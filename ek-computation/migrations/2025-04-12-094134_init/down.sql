-- This file should undo anything in `up.sql`
ALTER TABLE "instance"
DROP CONSTRAINT fk_instance_model;

ALTER TABLE "expert"
DROP CONSTRAINT fk_distribution_node;

ALTER TABLE "expert"
DROP CONSTRAINT fk_distribution_instance;

DROP TABLE IF EXISTS "model";

DROP TABLE IF EXISTS "node";

DROP TABLE IF EXISTS "expert";

DROP TABLE IF EXISTS "instance";