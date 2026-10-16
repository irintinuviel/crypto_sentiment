FROM jenkins/jenkins:lts-jdk17

USER root

# klient Dockera – Jenkins uruchamia kontenery treningowe i API na Dockerze hosta (docker.sock)
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker:27-cli /usr/local/libexec/docker/cli-plugins/docker-buildx /usr/local/libexec/docker/cli-plugins/docker-buildx

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN jenkins-plugin-cli --plugins \
    workflow-aggregator \
    git \
    github \
    docker-workflow \
    credentials-binding \
    timestamper \
    ws-cleanup
