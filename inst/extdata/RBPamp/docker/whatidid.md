# Build a docker image off of Ubuntu 16.04 LTS (works with singularity) from branch master

```
    docker build --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" --rm -t mjens:RBPamp .
```

# Build Cleanup
```
    docker rmi $(docker images -f "dangling=true" -q) --force
```

# Run it on sth

```
    docker run \
        --rm --user $(id -u):$(id -g) \
        -v ~/s2/yeolab/split_reads_F12WT_3ZF:/target \
        -t mjens:RBPamp \
        --opt-seed --run-path=1M --n-max=1000000 --n-sample=100000 --opt-nostruct --debug=opt
```

