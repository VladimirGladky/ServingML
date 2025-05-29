package client

import (
	"ServingML/gen/proto/model"
	"ServingML/pkg/logger"
	"context"
	"sync"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type Client struct {
	ctx  context.Context
	conn *grpc.ClientConn
}

func New(ctx context.Context) *Client {
	return &Client{
		ctx: ctx,
	}
}

func (c *Client) Run(wg *sync.WaitGroup, typeModel string) error {
	defer wg.Done()

	var err error
	c.conn, err = grpc.NewClient(
		"localhost:6060",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		logger.GetLoggerFromCtx(c.ctx).Error("connection failed", zap.Error(err))
		return err
	}
	defer c.conn.Close()

	client := model.NewBertServiceClient(c.conn)
	texts := []string{
		"What a great day!",
		"What a bad day!",
		"What a simple day",
	}

	for i := 0; i < 100; i++ {
		text := texts[i%len(texts)]

		reqCtx, cancel := context.WithTimeout(c.ctx, 5*time.Second)
		defer cancel()
		if typeModel == "firstmodel" {
			_, err := client.PredictFirstModel(reqCtx, &model.BertRequest{Text: text})
			if err != nil {
				logger.GetLoggerFromCtx(c.ctx).Error("predict failed",
					zap.String("text", text),
					zap.Error(err))
				continue
			}
		}
		if typeModel == "secondmodel" {
			_, err := client.PredictSecondModel(reqCtx, &model.BertRequest{Text: text})
			if err != nil {
				logger.GetLoggerFromCtx(c.ctx).Error("predict failed",
					zap.String("text", text),
					zap.Error(err))
				continue
			}
		}
	}
	return nil
}
