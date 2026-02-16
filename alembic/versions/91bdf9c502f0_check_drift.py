"""check drift

Revision ID: 91bdf9c502f0
Revises: c0ef3ff26306
Create Date: 2025-11-10 13:44:25.528310

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql
import pgvector
import pgvector.sqlalchemy
import letta.orm.custom_columns

# revision identifiers, used by Alembic.
revision: str = '91bdf9c502f0'
down_revision: Union[str, None] = 'c0ef3ff26306'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(inspector, name):
    return name in inspector.get_table_names()


def _column_exists(inspector, table, column):
    if not _table_exists(inspector, table):
        return False
    return any(c['name'] == column for c in inspector.get_columns(table))


def _index_exists(inspector, table, index_name):
    if not _table_exists(inspector, table):
        return False
    return any(i['name'] == index_name for i in inspector.get_indexes(table))


def _constraint_exists(bind, constraint_name):
    result = bind.execute(sa.text(
        "SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = :name"
    ), {"name": constraint_name})
    return result.fetchone() is not None


def upgrade() -> None:
    # This is a drift migration. The database may already have some or all of
    # these changes applied. Each operation is guarded with existence checks.
    bind = op.get_bind()
    insp = inspect(bind)

    # --- Create tables (only if they don't exist) ---
    if not _table_exists(insp, 'agent_passages'):
        op.create_table('agent_passages',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('text', sa.String(), nullable=False),
        sa.Column('embedding_config', letta.orm.custom_columns.EmbeddingConfigColumn(), nullable=False),
        sa.Column('metadata_', sa.JSON(), nullable=False),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(dim=4096), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), server_default=sa.text('FALSE'), nullable=False),
        sa.Column('_created_by_id', sa.String(), nullable=True),
        sa.Column('_last_updated_by_id', sa.String(), nullable=True),
        sa.Column('organization_id', sa.String(), nullable=False),
        sa.Column('agent_id', sa.String(), nullable=False),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
        sa.PrimaryKeyConstraint('id')
        )
    if not _index_exists(insp, 'agent_passages', 'agent_passages_created_at_id_idx'):
        op.create_index('agent_passages_created_at_id_idx', 'agent_passages', ['created_at', 'id'], unique=False)
    if not _index_exists(insp, 'agent_passages', 'agent_passages_org_idx'):
        op.create_index('agent_passages_org_idx', 'agent_passages', ['organization_id'], unique=False)
    if not _index_exists(insp, 'agent_passages', 'ix_agent_passages_org_agent'):
        op.create_index('ix_agent_passages_org_agent', 'agent_passages', ['organization_id', 'agent_id'], unique=False)

    if not _table_exists(insp, 'job_messages'):
        op.create_table('job_messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.String(), nullable=False),
        sa.Column('message_id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), server_default=sa.text('FALSE'), nullable=False),
        sa.Column('_created_by_id', sa.String(), nullable=True),
        sa.Column('_last_updated_by_id', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['message_id'], ['messages.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id', 'message_id', name='unique_job_message')
        )

    # --- Drop tables (only if they still exist) ---
    # Drop tables with foreign key dependencies first
    if _table_exists(insp, 'passage_tags'):
        if _index_exists(insp, 'passage_tags', 'ix_passage_tags_archive_id'):
            op.drop_index(op.f('ix_passage_tags_archive_id'), table_name='passage_tags')
        if _index_exists(insp, 'passage_tags', 'ix_passage_tags_archive_tag'):
            op.drop_index(op.f('ix_passage_tags_archive_tag'), table_name='passage_tags')
        if _index_exists(insp, 'passage_tags', 'ix_passage_tags_org_archive'):
            op.drop_index(op.f('ix_passage_tags_org_archive'), table_name='passage_tags')
        if _index_exists(insp, 'passage_tags', 'ix_passage_tags_tag'):
            op.drop_index(op.f('ix_passage_tags_tag'), table_name='passage_tags')
        op.drop_table('passage_tags')

    if _table_exists(insp, 'archival_passages'):
        if _index_exists(insp, 'archival_passages', 'archival_passages_created_at_id_idx'):
            op.drop_index(op.f('archival_passages_created_at_id_idx'), table_name='archival_passages')
        if _index_exists(insp, 'archival_passages', 'archival_passages_org_idx'):
            op.drop_index(op.f('archival_passages_org_idx'), table_name='archival_passages')
        if _index_exists(insp, 'archival_passages', 'ix_archival_passages_archive_id'):
            op.drop_index(op.f('ix_archival_passages_archive_id'), table_name='archival_passages')
        if _index_exists(insp, 'archival_passages', 'ix_archival_passages_org_archive'):
            op.drop_index(op.f('ix_archival_passages_org_archive'), table_name='archival_passages')
        op.drop_table('archival_passages')

    if _table_exists(insp, 'archives_agents'):
        op.drop_table('archives_agents')

    if _table_exists(insp, 'mcp_oauth'):
        op.drop_table('mcp_oauth')

    if _table_exists(insp, 'step_metrics'):
        op.drop_table('step_metrics')

    if _table_exists(insp, 'run_metrics'):
        op.drop_table('run_metrics')

    if _table_exists(insp, 'archives'):
        if _index_exists(insp, 'archives', 'ix_archives_created_at'):
            op.drop_index(op.f('ix_archives_created_at'), table_name='archives')
        if _index_exists(insp, 'archives', 'ix_archives_organization_id'):
            op.drop_index(op.f('ix_archives_organization_id'), table_name='archives')
        op.drop_table('archives')

    if _table_exists(insp, 'prompts'):
        op.drop_table('prompts')

    if _table_exists(insp, 'runs'):
        if _index_exists(insp, 'runs', 'ix_runs_agent_id'):
            op.drop_index(op.f('ix_runs_agent_id'), table_name='runs')
        if _index_exists(insp, 'runs', 'ix_runs_created_at'):
            op.drop_index(op.f('ix_runs_created_at'), table_name='runs')
        if _index_exists(insp, 'runs', 'ix_runs_organization_id'):
            op.drop_index(op.f('ix_runs_organization_id'), table_name='runs')
        op.drop_table('runs')

    if _table_exists(insp, 'mcp_tools'):
        op.drop_table('mcp_tools')

    # --- Drop columns (only if they still exist) ---
    if _column_exists(insp, 'agent_environment_variables', 'value_enc'):
        op.drop_column('agent_environment_variables', 'value_enc')

    if _index_exists(insp, 'agents', 'ix_agents_organization_id'):
        op.drop_index(op.f('ix_agents_organization_id'), table_name='agents')
    if _index_exists(insp, 'agents', 'ix_agents_organization_id_deployment_id'):
        op.drop_index(op.f('ix_agents_organization_id_deployment_id'), table_name='agents')
    if _index_exists(insp, 'agents', 'ix_agents_project_id'):
        op.drop_index(op.f('ix_agents_project_id'), table_name='agents')

    for col in ['timezone', 'entity_id', 'hidden', 'max_files_open',
                'per_file_view_window_char_limit', '_vector_db_namespace',
                'last_run_completion', 'deployment_id', 'last_run_duration_ms']:
        if _column_exists(insp, 'agents', col):
            op.drop_column('agents', col)

    if _index_exists(insp, 'agents_tags', 'ix_agents_tags_tag_agent_id'):
        op.drop_index(op.f('ix_agents_tags_tag_agent_id'), table_name='agents_tags')

    for idx in ['ix_block_hidden', 'ix_block_is_template', 'ix_block_label',
                'ix_block_org_project_template', 'ix_block_organization_id',
                'ix_block_organization_id_deployment_id', 'ix_block_project_id']:
        if _index_exists(insp, 'block', idx):
            op.drop_index(op.f(idx), table_name='block')

    for col in ['base_template_id', 'project_id', 'entity_id', 'hidden', 'template_id', 'deployment_id']:
        if _column_exists(insp, 'block', col):
            op.drop_column('block', col)

    if _index_exists(insp, 'blocks_agents', 'ix_blocks_agents_block_id'):
        op.drop_index(op.f('ix_blocks_agents_block_id'), table_name='blocks_agents')
    if _constraint_exists(bind, 'blocks_agents_agent_id_fkey'):
        op.drop_constraint(op.f('blocks_agents_agent_id_fkey'), 'blocks_agents', type_='foreignkey')
    if _constraint_exists(bind, 'fk_block_id_label'):
        op.drop_constraint(op.f('fk_block_id_label'), 'blocks_agents', type_='foreignkey')
    # Recreate foreign keys idempotently
    if not _constraint_exists(bind, 'fk_block_id_label'):
        op.create_foreign_key('fk_block_id_label', 'blocks_agents', 'block', ['block_id', 'block_label'], ['id', 'label'], initially='DEFERRED', deferrable=True)

    for col in ['original_file_name', 'total_chunks', 'chunks_embedded']:
        if _column_exists(insp, 'files', col):
            op.drop_column('files', col)

    for idx in ['ix_agent_filename', 'ix_file_agent', 'ix_files_agents_agent_id']:
        if _index_exists(insp, 'files_agents', idx):
            op.drop_index(op.f(idx), table_name='files_agents')
    if _constraint_exists(bind, 'uq_agent_filename'):
        op.drop_constraint(op.f('uq_agent_filename'), 'files_agents', type_='unique')
    if _constraint_exists(bind, 'uq_file_agent'):
        op.drop_constraint(op.f('uq_file_agent'), 'files_agents', type_='unique')
    if not _index_exists(insp, 'files_agents', 'ix_files_agents_agent_file_name'):
        op.create_index('ix_files_agents_agent_file_name', 'files_agents', ['agent_id', 'file_name'], unique=False)
    if not _index_exists(insp, 'files_agents', 'ix_files_agents_file_id_agent_id'):
        op.create_index('ix_files_agents_file_id_agent_id', 'files_agents', ['file_id', 'agent_id'], unique=False)
    if not _constraint_exists(bind, 'uq_files_agents_agent_file_name'):
        op.create_unique_constraint('uq_files_agents_agent_file_name', 'files_agents', ['agent_id', 'file_name'])
    if not _constraint_exists(bind, 'uq_files_agents_file_agent'):
        op.create_unique_constraint('uq_files_agents_file_agent', 'files_agents', ['file_id', 'agent_id'])
    if _constraint_exists(bind, 'files_agents_source_id_fkey'):
        op.drop_constraint(op.f('files_agents_source_id_fkey'), 'files_agents', type_='foreignkey')
    for col in ['source_id', 'start_line', 'end_line']:
        if _column_exists(insp, 'files_agents', col):
            op.drop_column('files_agents', col)

    for col in ['base_template_id', 'project_id', 'hidden', 'template_id', 'deployment_id']:
        if _column_exists(insp, 'groups', col):
            op.drop_column('groups', col)

    if _index_exists(insp, 'jobs', 'ix_jobs_user_id'):
        op.drop_index(op.f('ix_jobs_user_id'), table_name='jobs')
    if _constraint_exists(bind, 'fk_jobs_organization_id'):
        op.drop_constraint(op.f('fk_jobs_organization_id'), 'jobs', type_='foreignkey')
    for col in ['organization_id', 'ttft_ns', 'callback_error', 'total_duration_ns', 'background', 'stop_reason']:
        if _column_exists(insp, 'jobs', col):
            op.drop_column('jobs', col)

    for col in ['custom_headers_enc', 'token_enc', 'custom_headers']:
        if _column_exists(insp, 'mcp_server', col):
            op.drop_column('mcp_server', col)

    if _index_exists(insp, 'messages', 'ix_messages_run_id'):
        op.drop_index(op.f('ix_messages_run_id'), table_name='messages')
    if _index_exists(insp, 'messages', 'ix_messages_run_sequence'):
        op.drop_index(op.f('ix_messages_run_sequence'), table_name='messages')
    if _constraint_exists(bind, 'fk_messages_run_id'):
        op.drop_constraint(op.f('fk_messages_run_id'), 'messages', type_='foreignkey')
    for col in ['approve', 'denial_reason', 'is_err', 'approval_request_id', 'run_id', 'approvals']:
        if _column_exists(insp, 'messages', col):
            op.drop_column('messages', col)

    for col in ['access_key', 'region', 'api_version', 'access_key_enc', 'api_key_enc']:
        if _column_exists(insp, 'providers', col):
            op.drop_column('providers', col)

    if _column_exists(insp, 'sandbox_environment_variables', 'value_enc'):
        op.drop_column('sandbox_environment_variables', 'value_enc')

    if _index_exists(insp, 'source_passages', 'source_passages_file_id_idx'):
        op.drop_index(op.f('source_passages_file_id_idx'), table_name='source_passages')
    if _column_exists(insp, 'source_passages', 'tags'):
        op.drop_column('source_passages', 'tags')

    if _constraint_exists(bind, 'uq_source_name_organization'):
        op.drop_constraint(op.f('uq_source_name_organization'), 'sources', type_='unique')
    if _column_exists(insp, 'sources', 'vector_db_provider'):
        op.drop_column('sources', 'vector_db_provider')

    if _index_exists(insp, 'sources_agents', 'ix_sources_agents_source_id'):
        op.drop_index(op.f('ix_sources_agents_source_id'), table_name='sources_agents')

    if not _column_exists(insp, 'steps', 'job_id'):
        op.add_column('steps', sa.Column('job_id', sa.String(), nullable=True))
    if _index_exists(insp, 'steps', 'ix_steps_run_id'):
        op.drop_index(op.f('ix_steps_run_id'), table_name='steps')
    if _constraint_exists(bind, 'fk_steps_run_id'):
        op.drop_constraint(op.f('fk_steps_run_id'), 'steps', type_='foreignkey')
    for col in ['status', 'project_id', 'error_data', 'error_type', 'feedback', 'run_id', 'stop_reason']:
        if _column_exists(insp, 'steps', col):
            op.drop_column('steps', col)

    for col in ['default_requires_approval', 'enable_parallel_execution', 'npm_requirements']:
        if _column_exists(insp, 'tools', col):
            op.drop_column('tools', col)

    if _index_exists(insp, 'tools_agents', 'ix_tools_agents_tool_id'):
        op.drop_index(op.f('ix_tools_agents_tool_id'), table_name='tools_agents')


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_index(op.f('ix_tools_agents_tool_id'), 'tools_agents', ['tool_id'], unique=False)
    op.add_column('tools', sa.Column('npm_requirements', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True))
    op.add_column('tools', sa.Column('enable_parallel_execution', sa.BOOLEAN(), autoincrement=False, nullable=True))
    op.add_column('tools', sa.Column('default_requires_approval', sa.BOOLEAN(), autoincrement=False, nullable=True))
    op.add_column('steps', sa.Column('stop_reason', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('steps', sa.Column('run_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('steps', sa.Column('feedback', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('steps', sa.Column('error_type', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('steps', sa.Column('error_data', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True))
    op.add_column('steps', sa.Column('project_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('steps', sa.Column('status', postgresql.ENUM('PENDING', 'SUCCESS', 'FAILED', 'CANCELLED', name='stepstatus'), autoincrement=False, nullable=True))
    op.drop_constraint(None, 'steps', type_='foreignkey')
    op.create_foreign_key(op.f('fk_steps_run_id'), 'steps', 'runs', ['run_id'], ['id'], ondelete='SET NULL')
    op.create_index(op.f('ix_steps_run_id'), 'steps', ['run_id'], unique=False)
    op.drop_column('steps', 'job_id')
    op.create_index(op.f('ix_sources_agents_source_id'), 'sources_agents', ['source_id'], unique=False)
    op.add_column('sources', sa.Column('vector_db_provider', postgresql.ENUM('NATIVE', 'TPUF', 'PINECONE', name='vectordbprovider'), autoincrement=False, nullable=False))
    op.create_unique_constraint(op.f('uq_source_name_organization'), 'sources', ['name', 'organization_id'], postgresql_nulls_not_distinct=False)
    op.add_column('source_passages', sa.Column('tags', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True))
    op.create_index(op.f('source_passages_file_id_idx'), 'source_passages', ['file_id'], unique=False)
    op.add_column('sandbox_environment_variables', sa.Column('value_enc', sa.TEXT(), autoincrement=False, nullable=True))
    op.add_column('providers', sa.Column('api_key_enc', sa.TEXT(), autoincrement=False, nullable=True))
    op.add_column('providers', sa.Column('access_key_enc', sa.TEXT(), autoincrement=False, nullable=True))
    op.add_column('providers', sa.Column('api_version', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('providers', sa.Column('region', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('providers', sa.Column('access_key', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('messages', sa.Column('approvals', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True))
    op.add_column('messages', sa.Column('run_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('messages', sa.Column('approval_request_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('messages', sa.Column('is_err', sa.BOOLEAN(), autoincrement=False, nullable=True))
    op.add_column('messages', sa.Column('denial_reason', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('messages', sa.Column('approve', sa.BOOLEAN(), autoincrement=False, nullable=True))
    op.create_foreign_key(op.f('fk_messages_run_id'), 'messages', 'runs', ['run_id'], ['id'], ondelete='SET NULL')
    op.create_index(op.f('ix_messages_run_sequence'), 'messages', ['run_id', 'sequence_id'], unique=False)
    op.create_index(op.f('ix_messages_run_id'), 'messages', ['run_id'], unique=False)
    op.add_column('mcp_server', sa.Column('custom_headers', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True))
    op.add_column('mcp_server', sa.Column('token_enc', sa.TEXT(), autoincrement=False, nullable=True))
    op.add_column('mcp_server', sa.Column('custom_headers_enc', sa.TEXT(), autoincrement=False, nullable=True))
    op.add_column('jobs', sa.Column('stop_reason', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('jobs', sa.Column('background', sa.BOOLEAN(), autoincrement=False, nullable=True))
    op.add_column('jobs', sa.Column('total_duration_ns', sa.BIGINT(), autoincrement=False, nullable=True))
    op.add_column('jobs', sa.Column('callback_error', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('jobs', sa.Column('ttft_ns', sa.BIGINT(), autoincrement=False, nullable=True))
    op.add_column('jobs', sa.Column('organization_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.create_foreign_key(op.f('fk_jobs_organization_id'), 'jobs', 'organizations', ['organization_id'], ['id'])
    op.create_index(op.f('ix_jobs_user_id'), 'jobs', ['user_id'], unique=False)
    op.add_column('groups', sa.Column('deployment_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('groups', sa.Column('template_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('groups', sa.Column('hidden', sa.BOOLEAN(), autoincrement=False, nullable=True))
    op.add_column('groups', sa.Column('project_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('groups', sa.Column('base_template_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('files_agents', sa.Column('end_line', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('files_agents', sa.Column('start_line', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('files_agents', sa.Column('source_id', sa.VARCHAR(), autoincrement=False, nullable=False))
    op.create_foreign_key(op.f('files_agents_source_id_fkey'), 'files_agents', 'sources', ['source_id'], ['id'], ondelete='CASCADE')
    op.drop_constraint('uq_files_agents_file_agent', 'files_agents', type_='unique')
    op.drop_constraint('uq_files_agents_agent_file_name', 'files_agents', type_='unique')
    op.drop_index('ix_files_agents_file_id_agent_id', table_name='files_agents')
    op.drop_index('ix_files_agents_agent_file_name', table_name='files_agents')
    op.create_unique_constraint(op.f('uq_file_agent'), 'files_agents', ['file_id', 'agent_id'], postgresql_nulls_not_distinct=False)
    op.create_unique_constraint(op.f('uq_agent_filename'), 'files_agents', ['agent_id', 'file_name'], postgresql_nulls_not_distinct=False)
    op.create_index(op.f('ix_files_agents_agent_id'), 'files_agents', ['agent_id'], unique=False)
    op.create_index(op.f('ix_file_agent'), 'files_agents', ['file_id', 'agent_id'], unique=False)
    op.create_index(op.f('ix_agent_filename'), 'files_agents', ['agent_id', 'file_name'], unique=False)
    op.add_column('files', sa.Column('chunks_embedded', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('files', sa.Column('total_chunks', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('files', sa.Column('original_file_name', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.drop_constraint('fk_block_id_label', 'blocks_agents', type_='foreignkey')
    op.drop_constraint(None, 'blocks_agents', type_='foreignkey')
    op.create_foreign_key(op.f('fk_block_id_label'), 'blocks_agents', 'block', ['block_id', 'block_label'], ['id', 'label'], onupdate='CASCADE', ondelete='CASCADE', deferrable=True)
    op.create_foreign_key(op.f('blocks_agents_agent_id_fkey'), 'blocks_agents', 'agents', ['agent_id'], ['id'], ondelete='CASCADE')
    op.create_index(op.f('ix_blocks_agents_block_id'), 'blocks_agents', ['block_id'], unique=False)
    op.add_column('block', sa.Column('deployment_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('block', sa.Column('template_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('block', sa.Column('hidden', sa.BOOLEAN(), autoincrement=False, nullable=True))
    op.add_column('block', sa.Column('entity_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('block', sa.Column('project_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('block', sa.Column('base_template_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.create_index(op.f('ix_block_project_id'), 'block', ['project_id'], unique=False)
    op.create_index(op.f('ix_block_organization_id_deployment_id'), 'block', ['organization_id', 'deployment_id'], unique=False)
    op.create_index(op.f('ix_block_organization_id'), 'block', ['organization_id'], unique=False)
    op.create_index(op.f('ix_block_org_project_template'), 'block', ['organization_id', 'project_id', 'is_template'], unique=False)
    op.create_index(op.f('ix_block_label'), 'block', ['label'], unique=False)
    op.create_index(op.f('ix_block_is_template'), 'block', ['is_template'], unique=False)
    op.create_index(op.f('ix_block_hidden'), 'block', ['hidden'], unique=False)
    op.create_index(op.f('ix_agents_tags_tag_agent_id'), 'agents_tags', ['tag', 'agent_id'], unique=False)
    op.add_column('agents', sa.Column('last_run_duration_ms', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('agents', sa.Column('deployment_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('agents', sa.Column('last_run_completion', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True))
    op.add_column('agents', sa.Column('_vector_db_namespace', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('agents', sa.Column('per_file_view_window_char_limit', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('agents', sa.Column('max_files_open', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('agents', sa.Column('hidden', sa.BOOLEAN(), autoincrement=False, nullable=True))
    op.add_column('agents', sa.Column('entity_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('agents', sa.Column('timezone', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.create_index(op.f('ix_agents_project_id'), 'agents', ['project_id'], unique=False)
    op.create_index(op.f('ix_agents_organization_id_deployment_id'), 'agents', ['organization_id', 'deployment_id'], unique=False)
    op.create_index(op.f('ix_agents_organization_id'), 'agents', ['organization_id'], unique=False)
    op.add_column('agent_environment_variables', sa.Column('value_enc', sa.TEXT(), autoincrement=False, nullable=True))
    op.create_table('passage_tags',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('tag', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('passage_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('archive_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('is_deleted', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=False),
    sa.Column('_created_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('_last_updated_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('organization_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['archive_id'], ['archives.id'], name=op.f('passage_tags_archive_id_fkey'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name=op.f('passage_tags_organization_id_fkey')),
    sa.ForeignKeyConstraint(['passage_id'], ['archival_passages.id'], name=op.f('passage_tags_passage_id_fkey'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('passage_tags_pkey')),
    sa.UniqueConstraint('passage_id', 'tag', name=op.f('uq_passage_tag'), postgresql_include=[], postgresql_nulls_not_distinct=False)
    )
    op.create_index(op.f('ix_passage_tags_tag'), 'passage_tags', ['tag'], unique=False)
    op.create_index(op.f('ix_passage_tags_org_archive'), 'passage_tags', ['organization_id', 'archive_id'], unique=False)
    op.create_index(op.f('ix_passage_tags_archive_tag'), 'passage_tags', ['archive_id', 'tag'], unique=False)
    op.create_index(op.f('ix_passage_tags_archive_id'), 'passage_tags', ['archive_id'], unique=False)
    op.create_table('archival_passages',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('text', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('embedding_config', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=False),
    sa.Column('metadata_', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.Vector(dim=4096), autoincrement=False, nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('is_deleted', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=False),
    sa.Column('_created_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('_last_updated_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('organization_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('archive_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('tags', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['archive_id'], ['archives.id'], name=op.f('agent_passages_archive_id_fkey'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name=op.f('agent_passages_organization_id_fkey')),
    sa.PrimaryKeyConstraint('id', name=op.f('agent_passages_pkey'))
    )
    op.create_index(op.f('ix_archival_passages_org_archive'), 'archival_passages', ['organization_id', 'archive_id'], unique=False)
    op.create_index(op.f('ix_archival_passages_archive_id'), 'archival_passages', ['archive_id'], unique=False)
    op.create_index(op.f('archival_passages_org_idx'), 'archival_passages', ['organization_id'], unique=False)
    op.create_index(op.f('archival_passages_created_at_id_idx'), 'archival_passages', ['created_at', 'id'], unique=False)
    op.create_table('mcp_tools',
    sa.Column('mcp_server_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('tool_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('is_deleted', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=False),
    sa.Column('_created_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('_last_updated_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('organization_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name=op.f('mcp_tools_organization_id_fkey')),
    sa.PrimaryKeyConstraint('id', name=op.f('mcp_tools_pkey'))
    )
    op.create_table('runs',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('status', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('completed_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('stop_reason', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('background', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('metadata_', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True),
    sa.Column('request_config', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True),
    sa.Column('agent_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('callback_url', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('callback_sent_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('callback_status_code', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('callback_error', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('ttft_ns', sa.BIGINT(), autoincrement=False, nullable=True),
    sa.Column('total_duration_ns', sa.BIGINT(), autoincrement=False, nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('is_deleted', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=False),
    sa.Column('_created_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('_last_updated_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('organization_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('project_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('base_template_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('template_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('deployment_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], name='runs_agent_id_fkey'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name='runs_organization_id_fkey'),
    sa.PrimaryKeyConstraint('id', name='runs_pkey'),
    postgresql_ignore_search_path=False
    )
    op.create_index(op.f('ix_runs_organization_id'), 'runs', ['organization_id'], unique=False)
    op.create_index(op.f('ix_runs_created_at'), 'runs', ['created_at', 'id'], unique=False)
    op.create_index(op.f('ix_runs_agent_id'), 'runs', ['agent_id'], unique=False)
    op.create_table('prompts',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('prompt', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('is_deleted', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=False),
    sa.Column('_created_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('_last_updated_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('project_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('prompts_pkey'))
    )
    op.create_table('run_metrics',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('run_start_ns', sa.BIGINT(), autoincrement=False, nullable=True),
    sa.Column('run_ns', sa.BIGINT(), autoincrement=False, nullable=True),
    sa.Column('num_steps', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('is_deleted', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=False),
    sa.Column('_created_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('_last_updated_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('project_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('agent_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('organization_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('base_template_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('template_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('deployment_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('tools_used', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], name=op.f('run_metrics_agent_id_fkey'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['id'], ['runs.id'], name=op.f('run_metrics_id_fkey'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name=op.f('run_metrics_organization_id_fkey')),
    sa.PrimaryKeyConstraint('id', name=op.f('run_metrics_pkey'))
    )
    op.create_table('step_metrics',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('organization_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('provider_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('llm_request_ns', sa.BIGINT(), autoincrement=False, nullable=True),
    sa.Column('tool_execution_ns', sa.BIGINT(), autoincrement=False, nullable=True),
    sa.Column('step_ns', sa.BIGINT(), autoincrement=False, nullable=True),
    sa.Column('base_template_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('template_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('is_deleted', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=False),
    sa.Column('_created_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('_last_updated_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('project_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('agent_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('step_start_ns', sa.BIGINT(), autoincrement=False, nullable=True),
    sa.Column('llm_request_start_ns', sa.BIGINT(), autoincrement=False, nullable=True),
    sa.Column('run_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], name=op.f('step_metrics_agent_id_fkey'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['id'], ['steps.id'], name=op.f('step_metrics_id_fkey'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name=op.f('step_metrics_organization_id_fkey'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['provider_id'], ['providers.id'], name=op.f('step_metrics_provider_id_fkey'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['run_id'], ['runs.id'], name=op.f('fk_step_metrics_run_id'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('step_metrics_pkey'))
    )
    op.create_table('archives',
    sa.Column('name', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('description', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('metadata_', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True),
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=True),
    sa.Column('is_deleted', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=False),
    sa.Column('_created_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('_last_updated_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('organization_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('vector_db_provider', postgresql.ENUM('NATIVE', 'TPUF', 'PINECONE', name='vectordbprovider'), autoincrement=False, nullable=False),
    sa.Column('_vector_db_namespace', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('embedding_config', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name='archives_organization_id_fkey'),
    sa.PrimaryKeyConstraint('id', name='archives_pkey'),
    postgresql_ignore_search_path=False
    )
    op.create_index(op.f('ix_archives_organization_id'), 'archives', ['organization_id'], unique=False)
    op.create_index(op.f('ix_archives_created_at'), 'archives', ['created_at', 'id'], unique=False)
    op.create_table('archives_agents',
    sa.Column('agent_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('archive_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), autoincrement=False, nullable=False),
    sa.Column('is_owner', sa.BOOLEAN(), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], name=op.f('archives_agents_agent_id_fkey'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['archive_id'], ['archives.id'], name=op.f('archives_agents_archive_id_fkey'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('agent_id', 'archive_id', name=op.f('archives_agents_pkey')),
    sa.UniqueConstraint('agent_id', name=op.f('unique_agent_archive'), postgresql_include=[], postgresql_nulls_not_distinct=False)
    )
    op.create_table('mcp_oauth',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('state', sa.VARCHAR(length=255), autoincrement=False, nullable=False),
    sa.Column('server_id', sa.VARCHAR(length=255), autoincrement=False, nullable=True),
    sa.Column('server_url', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('server_name', sa.TEXT(), autoincrement=False, nullable=False),
    sa.Column('authorization_url', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('authorization_code', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('access_token', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('refresh_token', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('token_type', sa.VARCHAR(length=50), autoincrement=False, nullable=False),
    sa.Column('expires_at', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('scope', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('client_id', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('client_secret', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('redirect_uri', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('status', sa.VARCHAR(length=20), autoincrement=False, nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False),
    sa.Column('is_deleted', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=False),
    sa.Column('_created_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('_last_updated_by_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('organization_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('user_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('access_token_enc', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('refresh_token_enc', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('client_secret_enc', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('authorization_code_enc', sa.TEXT(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name=op.f('mcp_oauth_organization_id_fkey')),
    sa.ForeignKeyConstraint(['server_id'], ['mcp_server.id'], name=op.f('mcp_oauth_server_id_fkey'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('mcp_oauth_user_id_fkey')),
    sa.PrimaryKeyConstraint('id', name=op.f('mcp_oauth_pkey')),
    sa.UniqueConstraint('state', name=op.f('mcp_oauth_state_key'), postgresql_include=[], postgresql_nulls_not_distinct=False)
    )
    op.drop_table('job_messages')
    op.drop_index('ix_agent_passages_org_agent', table_name='agent_passages')
    op.drop_index('agent_passages_org_idx', table_name='agent_passages')
    op.drop_index('agent_passages_created_at_id_idx', table_name='agent_passages')
    op.drop_table('agent_passages')
    # ### end Alembic commands ###
