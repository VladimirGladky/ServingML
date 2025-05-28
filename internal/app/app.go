package app

import (
	grpcapp2 "ServingML/internal/app/grpc"
	service2 "ServingML/internal/batcher"
	"ServingML/internal/config"
	"ServingML/internal/inference"
	"ServingML/internal/modelWrapper"
	"ServingML/pkg/logger"
	"context"
	"os"
	"os/signal"
	"sync"
	"syscall"

	ort "github.com/yalue/onnxruntime_go"
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
	ort.SetSharedLibraryPath("/usr/lib/libonnxruntime.so")
	if err := ort.InitializeEnvironment(); err != nil {
		panic(err)
	}
	firstModel, err := modelWrapper.NewWrapperModel(
		cfg.Model1TokenizerPath,
		cfg.Model1Path,
		cfg.Model2BatchSize, cfg.Model1OutputSize)
	if err != nil {
		panic(err)
	}
	secondModel, err := modelWrapper.NewWrapperModel(
		cfg.Model1TokenizerPath,
		cfg.Model2Path,
		cfg.Model2BatchSize, cfg.Model2OutputSize)
	if err != nil {
		panic(err)
	}
	inf := inference.New()
	service := service2.New(ctx, firstModel, secondModel, inf)
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
