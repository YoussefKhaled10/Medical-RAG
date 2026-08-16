import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool


PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

# Look for .env in src/, root, or docker/
src_env = PROJECT_ROOT / "src" / ".env"
root_env = PROJECT_ROOT / ".env"
docker_env = PROJECT_ROOT / "docker" / ".env"
if src_env.exists():
    load_dotenv(src_env)
elif root_env.exists():
    load_dotenv(root_env)
elif docker_env.exists():
    load_dotenv(docker_env)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Importing the package registers every ORM model in SQLAlchemy metadata.
from src.models.db_schemes.medical_rag import SQLAlchemyBase


target_metadata = SQLAlchemyBase.metadata

database_url = os.getenv("POSTGRES_SYNC_URL")
if not database_url:
    raise RuntimeError(
        "POSTGRES_SYNC_URL is missing. Add it to .env or docker/.env."
    )

# Alembic ConfigParser treats percent signs as interpolation markers.
config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))


def run_migrations_offline() -> None:
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
