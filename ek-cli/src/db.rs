use clap::Subcommand;
use diesel::{Connection, PgConnection, RunQueryDsl, sql_query};
use diesel_migrations::{EmbeddedMigrations, MigrationHarness, embed_migrations};
use ek_base::{config::get_ek_settings, error::EKResult};

pub const MIGRATIONS: EmbeddedMigrations = embed_migrations!("../ek-computation/migrations/");

pub async fn drain_db(conn: &mut PgConnection) -> EKResult<()> {
    sql_query("DROP SCHEMA public CASCADE;")
        .execute(conn)
        .unwrap();

    sql_query("CREATE SCHEMA public;").execute(conn).unwrap();
    Ok(())
}

pub async fn run_migrations(conn: &mut PgConnection) -> EKResult<()> {
    conn.run_pending_migrations(MIGRATIONS).unwrap();
    log::info!("Migrations completed successfully");
    Ok(())
}

#[derive(Subcommand, Debug)]
pub enum DBCommand {
    #[command(about = "Run database migrations (create tables, etc.)")]
    Migrate,
    #[command(
        about = "Drop ALL the table and data in the database. This is a destructive operation. Use with caution."
    )]
    Drain,
}

pub async fn execute_db(cmd: DBCommand) -> EKResult<()> {
    let settings = get_ek_settings();
    let mut conn = PgConnection::establish(&settings.db_dsn).unwrap();

    match cmd {
        DBCommand::Migrate => run_migrations(&mut conn).await,
        DBCommand::Drain => drain_db(&mut conn).await,
    }
}
