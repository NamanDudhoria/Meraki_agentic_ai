-- Run this once in your Supabase SQL Editor before ingesting.

-- 1. Enable pgvector
create extension if not exists vector;

-- 2. Documents table
--    embedding dimension = 384 (sentence-transformers/all-MiniLM-L6-v2)
create table if not exists meraki_documents (
    id          text primary key,
    content     text not null,
    metadata    jsonb not null default '{}',
    embedding   vector(384) not null,
    updated_at  timestamptz not null default now()
);

-- 3. IVFFlat index for fast ANN search (rebuild after bulk ingest if needed)
create index if not exists meraki_documents_embedding_idx
    on meraki_documents
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- 4. Similarity search function used by agent.py
create or replace function match_meraki_documents(
    query_embedding vector(384),
    match_count     int,
    filter          jsonb default '{}'
)
returns table (
    id         text,
    content    text,
    metadata   jsonb,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        d.id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) as similarity
    from meraki_documents d
    where
        case
            when filter = '{}'::jsonb then true
            else d.metadata @> filter
        end
    order by d.embedding <=> query_embedding
    limit match_count;
end;
$$;
