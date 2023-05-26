# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

PRESIGNED_URL="https://dobf1k6cxlizq.cloudfront.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kb2JmMWs2Y3hsaXpxLmNsb3VkZnJvbnQubmV0LyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2ODU0NjA0MDN9fX1dfQ__&Signature=UqZJyn-jqctgWRfXBZjO-tvsghh33-M5lt5JI6El5m2zfwCp6ELM27T1bdqhfzN0yqvgVqEGcK0Z3~iry8HUZaiTz2wzgA-r2~EEe-n9qscUhAZgml~COIVOqu74hr7xdJobh6mn4VGAnzBtzO0Ds4x--A8MU9pwrS2f~P38oD3lT3I47fYZkfCUEK1~V9Uif8HzeF9kWs7YFDLCM7894Ts3YJ~N1HG~HMqcRDUnkav-6kSDH~KGJ1plIBydTV5dIHt1dCG49dNcBpGh3hAhNhEqi5mlh~ZshHXBQw5E0rCI~nu0SpRZU6rb9ixIOH51jFgNdyrlzXlodG1~zS54bQ__&Key-Pair-Id=K231VYXPC1TA1R"             
MODEL_SIZE="7B"                         # edit this list with the model sizes you wish to download
TARGET_FOLDER="/home/ubuntu/Models"      

declare -A N_SHARD_DICT

N_SHARD_DICT["7B"]="0"
N_SHARD_DICT["13B"]="1"
N_SHARD_DICT["30B"]="3"
N_SHARD_DICT["65B"]="7"

echo "Downloading tokenizer"
wget ${PRESIGNED_URL/'*'/"tokenizer.model"} -O ${TARGET_FOLDER}"/tokenizer.model"
wget ${PRESIGNED_URL/'*'/"tokenizer_checklist.chk"} -O ${TARGET_FOLDER}"/tokenizer_checklist.chk"

(cd ${TARGET_FOLDER} && md5sum -c tokenizer_checklist.chk)

for i in ${MODEL_SIZE//,/ }
do
    echo "Downloading ${i}"
    mkdir -p ${TARGET_FOLDER}"/${i}"
    for s in $(seq -f "0%g" 0 ${N_SHARD_DICT[$i]})
    do
        wget ${PRESIGNED_URL/'*'/"${i}/consolidated.${s}.pth"} -O ${TARGET_FOLDER}"/${i}/consolidated.${s}.pth"
    done
    wget ${PRESIGNED_URL/'*'/"${i}/params.json"} -O ${TARGET_FOLDER}"/${i}/params.json"
    wget ${PRESIGNED_URL/'*'/"${i}/checklist.chk"} -O ${TARGET_FOLDER}"/${i}/checklist.chk"
    echo "Checking checksums"
    (cd ${TARGET_FOLDER}"/${i}" && md5sum -c checklist.chk)
done