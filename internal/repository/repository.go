package repository

import (
	"context"
	"fmt"
	"github.com/jackc/pgx/v5"
)

type MLRepositoryInterface interface {
	SaveExpr(ctx context.Context, id, text string) error
}

type MLRepository struct {
	db  *pgx.Conn
	ctx context.Context
}

func (r *MLRepository) SaveExpr(ctx context.Context, id, text string) error {
	_, err := r.db.Exec(ctx,
		"INSERT INTO expressions (id, text,result) VALUES ($1, $2, $3)",
		id, text, "")
	if err != nil {
		return fmt.Errorf("unable to save expression: %w", err)
	}
	return nil
}
