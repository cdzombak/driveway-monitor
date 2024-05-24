# driveway-monitor

**Receive customizable, AI-powered notifications when someone arrives in your driveway.**

`driveway-monitor` accepts an RTSP video stream (or, for testing purposes, a video file) and uses the [YOLOv8 model](https://docs.ultralytics.com/models/yolov8/) to track objects in the video. The model can run on your CPU or on NVIDIA or Apple Silicon GPUs. When an object meets your notification criteria (highly customizable; see "Configuration" below), `driveway-monitor` will notify you via [Ntfy](https://ntfy.sh). The notification includes a snapshot of the object that triggered the notification and provides options to mute notifications for a period of time.

## Usage

```text
python3 main.py [-h] [--config CONFIG] [--video VIDEO] [--debug]
```

The `main.py` program only takes a few options on the CLI. Most configuration is done via a JSON config file (see "Configuration" below).

### Options

- `--config CONFIG`: Path to your JSON config file.
- `--debug`: Enable debug logging.
- `-h, --help`: Show help and exit.
- `--print`: Print notifications to stdout instead of sending them via Ntfy.
- `--video VIDEO`: Path to the video file or RTSP stream to process. _Required._

## Running via Docker

Due to the Python 3.12 requirement and the annoyance of maintaining a virtualenv, I recommend running this application via Docker. The following images are available:

- `cdzombak/driveway-monitor:*-amd64-cuda`: NVIDIA image for amd64 hosts
- `cdzombak/driveway-monitor:*-amd64-cpu`: CPU-only image for amd64 hosts

An Apple Silicon & Raspberry Pi compatible arm64 image (`cdzombak/driveway-monitor:arm64-*`) is planned (see issue # TK).

> [!NOTE]
> To run the model on an Apple Silicon GPU, you'll need to set up a Python virtualenv and run `driveway-monitor` directly, not via Docker. See "Running under Python/virtualenv" below.

Running a one-off `driveway-monitor` process with Docker might look like:

```shell
docker run --rm -v ./config.json:/config.json:ro cdzombak/driveway-monitor:1-amd64-cpu --config /config.json --video "rtsps://192.168.0.77:7441/abcdef?enableSrtp" --debug
```

### Sample `docker-compose.yml`

This `docker-compose.yml` file runs `driveway-monitor` on an amd64 host, with NVIDIA GPU support. Note that your config file is mapped into the container at `/config.json`.

```yaml
---
services:
  driveway-monitor:
    image: cdzombak/driveway-monitor:1-amd64-cuda
    volumes:
      - ./config.json:/config.json:ro
    command:
      [
        "--debug",
        "--config",
        "/config.json",
        "--video",
        "rtsp://192.168.0.77:7441/ca55e77e",
      ]
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

### Linux/NVIDIA GPU support under Docker

See [Ultralytics' docs on setting up Docker with NVIDIA support](https://docs.ultralytics.com/guides/docker-quickstart/#setting-up-docker-with-nvidia-support). In case that URL changes, the relevant instructions as of 2024-05-23 are copied here:

> First, verify that the NVIDIA drivers are properly installed by running:
>
> ```shell
> nvidia-smi
> ```
>
> Now, let's install the NVIDIA Docker runtime to enable GPU support in Docker containers:
>
> **Add NVIDIA package repositories**
>
> ```shell
> curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
> distribution=$(lsb_release -cs)
> curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
> ```
>
> **Install NVIDIA Docker runtime**
>
> ```shell
> sudo apt-get update
> sudo apt-get install -y nvidia-docker2
> ```
>
> **Restart Docker service to apply changes**
>
> ```shell
> sudo systemctl restart docker
> ```
>
> **Verify NVIDIA Runtime with Docker**
>
> Run `docker info | grep -i runtime` to ensure that nvidia appears in the list of runtimes.

## Running with Python (in a virtualenv)

> [!NOTE]
> Requires Python 3.12 or later.

Clone the repository, change into its directory, set up a virtualenv with the project's requirements, and run `main.py`:

```shell
git clone https://github.com/cdzombak/driveway-monitor.git
cd driveway-monitor
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
python3 ./main.py --config /path/to/config.json --video 'rtsp://example.com/mystream'
```

If you're running on Apple Silicon, you should see a log message at startup informing you the model is using the `mps` device.

## Overview of Operation

This section briefly explains the different components of the program. In particular, this understanding will help you effectively configure `driveway-monitor`.

### Prediction Model

(Configuration key: `model`.)

The prediction model consumes a video stream frame-by-frame and feeds each frame to the YOLOv8 model. The model produces predictions of objects in the frame, including their classifications (e.g. "car") and rectangular bounding boxes. These predictions are passed to the tracker process.

### Tracker

(Configuration keys: `tracker` and `notification_criteria`.)

The tracker process aggregates the model's predictions over time, building tracks that represent the movement over time of individual objects in the video stream. Every time a track is updated with a prediction from a new frame, the tracker evaluates the track against the notification criteria. If the track meets the criteria, a notification is triggered.

### Notifier

(Configuration key: `notifier`.)

The notifier receives notification triggers from the tracker process and sends via [Ntfy](https://ntfy.sh) to a configured server and topic. Notifications include a snapshot of the object that triggered the notification and provides options to mute notifications for a period of time.

The notifier also debounces notifications, preventing multiple notifications for the same type of object within a short time period; and allows muting all notifications for a period of time.

### Web Server

(Configuration key: `web`.)

The web server provides a few simple endpoints for viewing notification photos, muting `driveway-monitor`'s notifications, and checking the program's health.

## Configuration (`config.json`)

Quite a number of parameters are set by a JSON configuration file. An example is provided in this repo ([`config.example.json`](config.example.json)) which demonstrates most, but not all, these options.

The file is a single JSON object containing the following keys, or a subset thereof. All keys in the JSON file are optional; if a key is not present, the default value will be used. Each key refers to another object configuring a specific part of the `driveway-monitor` program:

- `model`: Configures video capture and the AI model. (See [Predict Settings](https://docs.ultralytics.com/usage/cfg/#predict-settings).)
  - `confidence`: Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.
  - `device`: Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`).
  - `half`: Use half precision (FP16) to speed up inference.
  - `healthcheck_ping_url`: URL to ping with a GET request when the program starts and every `liveness_tick_s` seconds.
  - `iou`: Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.
  - `liveness_tick_s`: Specifies the interval to log a liveness message and ping `healthcheck_ping_url` (if that field is set).
  - `max_det`: Maximum number of detections per frame.
- `tracker`: Configures the system that builds tracks from the model's detections over time.
  - `inactive_track_prune_s`: Specifies the number of seconds after which an inactive track is pruned. This prevents incorrectly adding a new prediction to an old track.
  - `track_connect_min_overlap`: Minimum overlap percentage of a prediction box with the average of the last 2 boxes in an existing track for the prediction to be added to that track.
- `notifier`: Configures how notifications are sent.
  - `debounce_threshold_s`: Specifies the number of seconds to wait after a notification before sending another one for the same type of object.
  - `default_priority`: Default priority for notifications. ([See Ntfy docs on Message Priority](https://docs.ntfy.sh/publish/#message-priority).)
  - `priorities`: Map of `classification name` → `priority`. Allows customizing notification priority for specific object types.
  - `req_timeout_s`: Request timeout for sending notifications.
  - `server`: The Ntfy server to send notifications to.
  - `token`: Ntfy auth token (beginning with `tk_`).
  - `topic`: Ntfy topic to send to.
- `notification_criteria`: Configures the criteria for sending notifications.
  - `classification_allowlist`: List of object classifications to allow. If this list is non-empty, only objects with classifications in this list will be considered for notifications.
  - `classification_blocklist`: List of object classifications to block. Objects with classifications in this list will not be considered for notifications.
  - `min_track_length_s`: Minimum number of seconds a track must cover before it can trigger a notification.
  - `min_track_length_s_per_classification`: Map of `classification name` → `seconds`. Allows customizing `min_track_length_s` on a per-classification basis.
  - `track_cel`: [CEL](https://cel.dev) expression to evaluate for each track. If the expression evaluates to `true`, a notification will be sent. (See "The `notification_criteria.track_cel.track_cel` expression" below.)
- `web`: Configures the embedded web server.
  - `bind_to`: IP address to bind the web server to.
  - `external_base_url`: External base URL for the web server (e.g. `http://me.example-tailnet.ts.net:5550`). Used to generate URLs in notifications.
  - `port`: Port to bind the web server to.

### The `notification_criteria.track_cel.track_cel` expression

This field is a string representing a [CEL](https://cel.dev) expression that will be evaluated for each track. If the expression evaluates to `true`, a notification will be sent.

The expression has access to a single variable, `track`. This variable is a CEL map with the following keys:

- `classification` (string): classification of the object in the track.
- `predictions`: list of `Prediction`s from the model included in this track, each of which have:
  - `box`: the `Box` for this prediction
  - `classification` (string): classification of the object
  - `t` timestamp
- `first_t`: timestamp of the first prediction in the track
- `last_t`: timestamp of the last prediction in the track
- `length_t`: duration of the track, in seconds (this is just `last_t - first_t`)
- `first_box`: the first prediction's `Box`
- `last_box`: the last prediction's `Box`
- `average_box`: the average of the first and last prediction's `Box`
- `total_box`: the smallest `Box` that covers all predictions in the track

Each `Box` has:

- `a`: top-left `Point`
- `b`: bottom-right `Point`
- `center`: the center `Point`
- `w`: width of the box
- `h`: height of the box
- `area`: area of the box

Finally, a `Point` has:

- `x`: x-coordinate
- `y`: y-coordinate

#### Box coordinates

Coordinates are floats between 0 and 1, on both axes.

The origin for box coordinates (`(0.0, 0.0)`) is the top-left corner of the frame. Coordinate `(1, 1)` is the lower-right corner of the frame.

#### Movement vector

A track's movement vector is a vector, from the center of the prediction box at the start of the track, to the center of the most recent prediction box in the track. It has two properties, `length` and `direction`.

- `length`: The length of the vector, as a float between 0 and 1.
- `direction`: The direction of the vector, in degrees, from `[-180, 180)`. 0° is straight to the right of frame; 90º is straight up; -180º is straight left; -90º is straight down.

> [!NOTE]
> The `direction` thing is kind of awkward here and there's room for improvement. I hacked this together really quickly 🤷🏻‍♂️

```text
        ┌───┐    ┌───┐    ┌───┐
        │ b │    │ b │    │ b │
        └─▲─┘    └─▲─┘    └─▲─┘
           ╲       │       ╱
          135º   90º    45º
             ╲     │     ╱
              ╲    │    ╱
               ╳───┴───╳
 ┌───┐         │       │         ┌───┐
 │ b ◀─-180º───┤   a   │───0º────▶ b │
 └───┘         │       │         └───┘
               ╳───┬───╳
              ╱    │    ╲
             ╱   -90º    ╲
         -135º     │    -45º
           ╱       │       ╲
        ┌─▼─┐    ┌─▼─┐    ┌─▼─┐
        │ b │    │ b │    │ b │
        └───┘    └───┘    └───┘
```

#### Example

Here's an example expression that I use at home:

```text
track.last_box.b.y > 0.4 && track.movement_vector.length > 0.4 && track.movement_vector.direction < 25 && track.movement_vector.direction > -80
```

This example limits notifications to tracks that meet all the following criteria:

- The most recent prediction box's bottom-right corner is in the lower 60% of the frame (i.e. that corner's Y coordinate is greater than 0.4).
- The vector from start to end of the track has moved across at least 40% of the frame.
- The direction of the movement vector is between 25° and -80° (down and to the right).

## Note on Ntfy, HTTPS, and Tailscale

The Ntfy web interface and iPhone app (maybe other clients, too, I'm not sure) will not send POST requests to insecure, `http://` endpoints. I highly recommend running `driveway-monitor` behind a reverse proxy with HTTPS.

To make `drivewway-monitor`'s API securely accessible wherever you are _and_ provide an HTTPS endpoint, I use and recommend [Tailscale](https://tailscale.com). On the machine running `driveway-monitor`, you can use [a single Tailscale command](https://tailscale.com/kb/1242/tailscale-serve) to make the API available over HTTPS. In this example, I have `driveway-monitor` running on `my-machine` and its API listening at port 5550 (the default). I'll tell Tailscale to make the API available over HTTPS at `https://my-machine.my-tailnet.ts.net:5559`:

```shell
tailscale serve --bg --tls-terminated-tcp 5559 tcp://localhost:5550
```

## Monitoring

`driveway-monitor` is designed to facilite monitoring via, for example, [Uptime Kuma](https://github.com/louislam/uptime-kuma). I recommend monitoring two things: the model's liveness and the API's health endpoint.

You could also monitor that the Docker container is running (or that the Python script is running, under systemd or similar), but that's less valuable.

### Model Liveness

If `model.healthcheck_ping_url` is set in your config, the model will send a GET request to `model.healthcheck_ping_url` every `model.liveness_tick_s` seconds. You can use this in conjunction with an Uptime Kuma "push" monitor to be alerted when the model stops pinging the URL.

### API Health

GET the `/health` endpoint on the web server (`web.external_base_url/health`). This endpoint returns an HTTP 200 with the JSON content `{"status": "ok"}` as long as the server is running.

## License

GNU GPL v3; see [LICENSE](LICENSE) in this repo.

## Author

Chris Dzombak.

- [dzombak.com](https://www.dzombak.com)
- [GitHub: @cdzombak](https://github.com/cdzombak)

## See Also

- [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [CEL](https://cel.dev)
- [Ntfy](https://ntfy.sh)
- [Uptime Kuma](https://github.com/louislam/uptime-kuma)
- [Tailscale](https://tailscale.com)