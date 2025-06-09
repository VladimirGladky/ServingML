package app

import (
	grpcapp2 "ServingML/internal/app/grpc"
	service2 "ServingML/internal/batcher"
	"ServingML/internal/config"
	"ServingML/internal/inference"
	"ServingML/pkg/logger"
	"context"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"go.uber.org/zap"
)

type App struct {
	cfg        *config.Config
	ctx        context.Context
	gRPCServer *grpcapp2.App
	wg         sync.WaitGroup
	cancel     context.CancelFunc
	service    *service2.ServiceBatcher
}

func New(cfg *config.Config, ctx context.Context) *App {
	batchSizes := make(map[string]int)
	for _, model := range cfg.Models {
		batchSizes[model.Name] = model.BatchSize
	}
	inf := inference.New(cfg)
	service := service2.New(ctx, inf, batchSizes, cfg)
	gRPCapp := grpcapp2.New(cfg, service, ctx)
	return &App{
		cfg:        cfg,
		ctx:        ctx,
		gRPCServer: gRPCapp,
		service:    service,
	}
}

func (a *App) MustRun() {
	if err := a.Run(); err != nil {
		panic(err)
	}
}

func (a *App) Run() error {
	errCh := make(chan error, 1)
	a.wg.Add(1)
	go func() {
		logger.GetLoggerFromCtx(a.ctx).Info("starting grpc server")
		defer a.wg.Done()
		if err := a.gRPCServer.Run(); err != nil {
			errCh <- err
			a.cancel()
		}
	}()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	select {
	case err := <-errCh:
		logger.GetLoggerFromCtx(a.ctx).Error("error running app", zap.Error(err))
		return err
	case sig := <-sigCh:
		logger.GetLoggerFromCtx(a.ctx).Info("received signal", zap.String("signal", sig.String()))
		a.Stop()
	case <-a.ctx.Done():
		logger.GetLoggerFromCtx(a.ctx).Info("context done")
	}

	return nil
}

func (a *App) Stop() {
	logger.GetLoggerFromCtx(a.ctx).Info("stopping app")
	a.cancel()
	a.gRPCServer.GRPCSrv.GracefulStop()
	a.wg.Wait()
	logger.GetLoggerFromCtx(a.ctx).Info("app stopped")
}
