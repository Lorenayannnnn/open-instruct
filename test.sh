BEAKER_VERSION=v1.5.235
curl --silent --connect-timeout 5 \
  --max-time 10 \
  --retry 5 \
  --retry-delay 0 \
  --retry-max-time 40 \
  --output beaker.tar.gz \
  "https://beaker.org/api/v3/release/cli?os=linux&arch=amd64&version=${BEAKER_VERSION}" \
  && tar -zxf beaker.tar.gz -C ~/.local/bin ./beaker \
  && rm beaker.tar.gz