synda_conf_dir=$ST_HOME/conf

if [[ -d "$synda_conf_dir" ]]
then
    echo "$synda_conf_dir exists on your filesystem."
else
    mkdir $synda_conf_dir
fi
    

cat << EOF > $ST_HOME/conf/credentials.conf
[esgf_credential]
openid = $OPEN_ID
password = $OPEN_ID_KEY
EOF