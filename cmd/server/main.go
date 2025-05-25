package main

import (
	"ServingML/internal/app"
	"ServingML/internal/config"
	"ServingML/pkg/logger"
	"context"
	"go.uber.org/zap"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	ctx, _ = logger.New(ctx)
	defer cancel()
	cfg, err := config.New()
	if err != nil {
		logger.GetLoggerFromCtx(ctx).Fatal("error loading config", zap.Error(err))
	}
	application := app.New(cfg, ctx)
	application.MustRun()
}
