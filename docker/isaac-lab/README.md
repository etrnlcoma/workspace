# Isaac Lab (Official Docker Flow)

This workspace uses the official Isaac Lab Docker workflow (option 2):

- Source of truth: `docker/isaac-lab/upstream/docker/*`
- Entrypoint: `docker/isaac-lab/upstream/docker/container.py`
- Compose file: `docker/isaac-lab/upstream/docker/docker-compose.yaml`
- Base env: `docker/isaac-lab/upstream/docker/.env.base`

## Recommended Commands

Run these from anywhere in the workspace:

```bash
/home/uzer/workspace/docker/isaac-lab/ctl.sh build base
/home/uzer/workspace/docker/isaac-lab/ctl.sh start base
/home/uzer/workspace/docker/isaac-lab/ctl.sh enter base
/home/uzer/workspace/docker/isaac-lab/ctl.sh stop base
```

## Notes

- This path does not use local custom Dockerfiles in `docker/isaac-lab`.
- If needed, adjust Isaac Sim image/version in:
  - `/home/uzer/workspace/docker/isaac-lab/upstream/docker/.env.base`
