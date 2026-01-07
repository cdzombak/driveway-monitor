# driveway-monitor

**Receive customizable, AI-powered notifications when someone arrives in your driveway.**

`driveway-monitor` accepts an RTSP video stream and uses the [YOLO11 model](https://docs.ultralytics.com/models/yolo11/) to track objects. When an object meets your notification criteria, it notifies you via [Ntfy](https://ntfy.sh) with a snapshot image.

## Images

- `cdzombak/driveway-monitor:VERSION-amd64`: For amd64 hosts (supports CUDA and CPU; auto-detects at runtime)
- `cdzombak/driveway-monitor:VERSION-arm64`: For arm64 hosts (Raspberry Pi 4/5, etc.)

Replace `VERSION` with a specific version (e.g., `1.0.0`) or `1` for the latest v1.x release.

## Quick Start

```shell
docker run --rm \
  -v ./config.json:/config.json:ro \
  cdzombak/driveway-monitor:1-amd64 \
  --config /config.json \
  --video "rtsp://192.168.0.77:7441/your-stream"
```

## Docker Compose (with NVIDIA GPU)

```yaml
services:
  driveway-monitor:
    image: cdzombak/driveway-monitor:1-amd64
    volumes:
      - ./config.json:/config.json:ro
      - ./enrichment-prompts:/enrichment-prompts:ro  # optional, for AI enrichment
    command: ["--config", "/config.json", "--video", "rtsp://192.168.0.77:7441/your-stream"]
    ports:
      - 5550:5550
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              count: all
    restart: always
```

## Configuration

See the [GitHub repository](https://github.com/cdzombak/driveway-monitor) for complete configuration documentation, including:

- Notification criteria and filtering
- CEL expressions for advanced track filtering
- Ollama/OpenAI enrichment setup
- Health monitoring integration

## License

GNU GPL v3. See [LICENSE](https://github.com/cdzombak/driveway-monitor/blob/main/LICENSE).

## Author

[Chris Dzombak](https://www.dzombak.com) · [GitHub](https://github.com/cdzombak)
