-- Your SQL goes here
CREATE TABLE
	"model" (
		"id" serial PRIMARY KEY,
		"name" TEXT NOT NULL,
		"config" JSONB NOT NULL
	);

CREATE TABLE
	"node" (
		"id" serial PRIMARY KEY,
		"hostname" TEXT NOT NULL,
		"device" TEXT NOT NULL,
		"config" JSONB NOT NULL
	);

CREATE TABLE
	"expert" (
		"id" serial PRIMARY KEY,
		"instance_id" INTEGER NOT NULL,
		"node_id" INTEGER NOT NULL,
		"expert_id" TEXT NOT NULL,
		"replica" INTEGER NOT NULL,
		"state" JSONB NOT NULL
	);

CREATE TABLE
	"instance" (
		"id" serial PRIMARY KEY,
		"model_id" INTEGER NOT NULL,
		"name" TEXT NOT NULL
	);

-- add foreign key between instance and  model on model_id
ALTER TABLE "instance" ADD CONSTRAINT fk_instance_model FOREIGN KEY (model_id) REFERENCES model (id) ON DELETE CASCADE;

ALTER TABLE "expert" ADD CONSTRAINT fk_distribution_node FOREIGN KEY (node_id) REFERENCES node (id) ON DELETE CASCADE;

ALTER TABLE "expert" ADD CONSTRAINT fk_distribution_instance FOREIGN KEY (instance_id) REFERENCES instance (id) ON DELETE CASCADE;

create UNIQUE INDEX  uniq_instance_name ON instance(name);
create UNIQUE INDEX  uniq_model_name ON model(name);
create UNIQUE INDEX  uniq_expert_instance_node ON expert(instance_id,node_id,expert_id);



CREATE UNIQUE INDEX "idx_node_hostname" ON "node" (hostname);