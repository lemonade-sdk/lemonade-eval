"""Initial database schema creation.

Revision ID: initial
Create Date: 2026-03-29

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema."""

    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=True),
        sa.Column('hashed_password', sa.String(length=255), nullable=True),
        sa.Column('api_key_hash', sa.String(length=255), nullable=True),
        sa.Column('api_key_prefix', sa.String(length=10), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_users_email', 'users', ['email'], unique=False)
    op.create_index('idx_users_api_key_prefix', 'users', ['api_key_prefix'], unique=False)

    # Create models table
    op.create_table('models',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('checkpoint', sa.String(length=500), nullable=False),
        sa.Column('model_type', sa.String(length=50), nullable=True),
        sa.Column('family', sa.String(length=100), nullable=True),
        sa.Column('parameters', sa.Integer(), nullable=True),
        sa.Column('max_context_length', sa.Integer(), nullable=True),
        sa.Column('architecture', sa.String(length=100), nullable=True),
        sa.Column('license_type', sa.String(length=100), nullable=True),
        sa.Column('hf_repo', sa.String(length=255), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('checkpoint')
    )
    op.create_index('idx_models_name', 'models', ['name'], unique=False)
    op.create_index('idx_models_family', 'models', ['family'], unique=False)
    op.create_index('idx_models_checkpoint', 'models', ['checkpoint'], unique=False)

    # Create model_versions table
    op.create_table('model_versions',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('model_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('quantization', sa.String(length=50), nullable=True),
        sa.Column('dtype', sa.String(length=50), nullable=True),
        sa.Column('backend', sa.String(length=100), nullable=True),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('is_default', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['model_id'], ['models.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_id', 'version', 'quantization', 'dtype', 'backend')
    )
    op.create_index('idx_model_versions_model', 'model_versions', ['model_id'], unique=False)

    # Create runs table
    op.create_table('runs',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('model_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('build_name', sa.String(length=255), nullable=False),
        sa.Column('run_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('status_message', sa.Text(), nullable=True),
        sa.Column('device', sa.String(length=50), nullable=True),
        sa.Column('backend', sa.String(length=100), nullable=True),
        sa.Column('dtype', sa.String(length=50), nullable=True),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Numeric(), nullable=True),
        sa.Column('system_info', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('lemonade_version', sa.String(length=20), nullable=True),
        sa.Column('build_uid', sa.String(length=100), nullable=True),
        sa.Column('log_file_path', sa.String(length=500), nullable=True),
        sa.Column('error_log', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['model_id'], ['models.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_runs_model', 'runs', ['model_id'], unique=False)
    op.create_index('idx_runs_user', 'runs', ['user_id'], unique=False)
    op.create_index('idx_runs_status', 'runs', ['status'], unique=False)
    op.create_index('idx_runs_type', 'runs', ['run_type'], unique=False)
    op.create_index('idx_runs_created', 'runs', ['created_at'], unique=False)
    op.create_index('idx_runs_device_dtype', 'runs', ['device', 'dtype'], unique=False)

    # Create metrics table
    op.create_table('metrics',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('display_name', sa.String(length=255), nullable=True),
        sa.Column('value_numeric', sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column('value_text', sa.Text(), nullable=True),
        sa.Column('unit', sa.String(length=50), nullable=True),
        sa.Column('mean_value', sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column('std_dev', sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column('min_value', sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column('max_value', sa.Numeric(precision=20, scale=6), nullable=True),
        sa.Column('iteration_values', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('run_id', 'category', 'name')
    )
    op.create_index('idx_metrics_run', 'metrics', ['run_id'], unique=False)
    op.create_index('idx_metrics_category', 'metrics', ['category'], unique=False)
    op.create_index('idx_metrics_name', 'metrics', ['name'], unique=False)
    op.create_index('idx_metrics_value', 'metrics', ['value_numeric'], unique=False)

    # Create tags table
    op.create_table('tags',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('color', sa.String(length=7), nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=False), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )

    # Create run_tags table (many-to-many)
    op.create_table('run_tags',
        sa.Column('run_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('tag_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['tag_id'], ['tags.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('run_id', 'tag_id')
    )


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table('run_tags')
    op.drop_table('tags')
    op.drop_table('metrics')
    op.drop_table('runs')
    op.drop_table('model_versions')
    op.drop_table('models')
    op.drop_table('users')
