package main

import (
	"ServingML/internal/client"
	"ServingML/pkg/logger"
	"context"
	"sync"
	"time"

	"go.uber.org/zap"
)

func main() {
	ctx := context.Background()
	ctx, _ = logger.New(ctx)
	var wg sync.WaitGroup
	start := time.Now()
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			client := client.New(ctx)
			err := client.Run(&wg, "russian-sentiment")
			if err != nil {
				logger.GetLoggerFromCtx(ctx).Fatal("error:", zap.Error(err))
			}
		}()
	}
	wg.Wait()
	elapsed := time.Since(start)
	logger.GetLoggerFromCtx(ctx).Info("elapsed first model", zap.Duration("elapsed", elapsed))

	start = time.Now()
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			client := client.New(ctx)
			err := client.Run(&wg, "emotion-detection")
			if err != nil {
				logger.GetLoggerFromCtx(ctx).Fatal("error:", zap.Error(err))
			}
		}()
	}
	wg.Wait()
	elapsed = time.Since(start)
	logger.GetLoggerFromCtx(ctx).Info("elapsed second model", zap.Duration("elapsed", elapsed))
}
