from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str
    port: int
    username: str
    password: str
    database: str
    secure: bool = True

    @classmethod
    def from_env(cls, database: str, prefix: str = "CH") -> ClickHouseConfig:
        return cls(
            host=os.environ.get(f"{prefix}_HOST", "ch.bloomsburytech.com"),
            port=int(os.environ.get(f"{prefix}_PORT", "443")),
            username=os.environ.get(f"{prefix}_USER", "default"),
            password=os.environ[f"{prefix}_PASSWORD"],
            database=os.environ.get(f"{prefix}_DATABASE", database),
            secure=os.environ.get(f"{prefix}_SECURE", "1") == "1",
        )


def get_client(config: ClickHouseConfig):
    import clickhouse_connect
    return clickhouse_connect.get_client(
        host=config.host,
        port=config.port,
        username=config.username,
        password=config.password,
        secure=config.secure,
        database=config.database,
    )
