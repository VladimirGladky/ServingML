package main

import (
	"ServingML/internal/client"
	"ServingML/pkg/logger"
	"context"
	"go.uber.org/zap"
	"sync"
	"time"
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
			client.Run(&wg)
		}()
	}
	wg.Wait()
	elapsed := time.Since(start)
	logger.GetLoggerFromCtx(context.Background()).Info("elapsed", zap.Duration("elapsed", elapsed))
}
