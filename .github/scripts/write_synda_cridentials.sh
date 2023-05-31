#!/bin/sh
bash -c "yes || true" | synda init-env
synda_conf_dir=$ST_HOME/conf

if [[ -d "$synda_conf_dir" ]]
then
    echo "$synda_conf_dir exists on your filesystem."
else
    mkdir $synda_conf_dir
fi

conf_file=$ST_HOME/conf/credentials.conf
cat << EOF > $conf_file
[esgf_credential]
openid = $OPEN_ID
password = $OPEN_ID_KEY
EOF

echo "$conf_file set"
cat $conf_file
