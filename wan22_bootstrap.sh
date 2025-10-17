name: Build and Push Docker Image

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      # Free up disk space on GitHub runner
      - name: Maximize build space
        run: |
          echo "Before cleanup:"
          df -h
          sudo rm -rf /usr/local/lib/android || true
          sudo rm -rf /opt/ghc || true
          sudo rm -rf /usr/share/dotnet || true
          sudo apt-get clean || true
          sudo rm -rf /var/lib/apt/lists/* || true
          sudo docker system prune -af || true
          echo "After cleanup:"
          df -h

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to DockerHub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      # Build & load locally so we can inspect before pushing
      - name: Build (load for inspection)
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ./Dockerfile
          load: true
          tags: test/wan22:inspect
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Inspect image for bootstrap
        run: |
          echo "Checking that /usr/local/bin/wan22_bootstrap.sh exists..."
          docker run --rm --entrypoint bash test/wan22:inspect -lc "
            set -e;
            ls -l /usr/local/bin/wan22_bootstrap.sh || (echo 'FATAL: bootstrap missing in image' && exit 1);
            echo '--- /workspace (first 120 lines) ---';
            ls -la /workspace | sed -n '1,120p';
          "

      # If inspection passes, push the final image
      - name: Build and Push (final)
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: |
            ${{ secrets.DOCKERHUB_USERNAME }}/wan22:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
