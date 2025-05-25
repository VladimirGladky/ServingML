package config

import (
	"ServingML/pkg/postgres"
	"github.com/ilyakaznacheev/cleanenv"
	"github.com/joho/godotenv"
)

type Config struct {
	Postgres postgres.Config `yaml:"Postgres" env:"POSTGRES"`
	GrpcPort string          `yaml:"grpc_port" env:"GRPC_PORT" env-default:"6060"`
	GrpcHost string          `yaml:"grpc_host" env:"GRPC_HOST" env-default:"localhost"`
}

func New() (*Config, error) {
	_ = godotenv.Load(".env")
	var cfg Config
	if err := cleanenv.ReadConfig("./config/config.yaml", &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}
