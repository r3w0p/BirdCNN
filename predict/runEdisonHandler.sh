#!/bin/bash
shopt -s nullglob

edison=./edison
uploads=${edison}/uploads
predictions=${edison}/predictions
temp=${edison}/preprocessed

mkdir -p ${edison} ${uploads} ${predictions} ${temp}

echo
echo "Listening for new .wav files..."

while true; do
  for file in ${uploads}/*.wav; do
    basename=`basename ${file}`                     # $file => example.wav
    filename=`echo "${basename}" | cut -f 1 -d '.'` # example.wav => example
    tempfile=${temp}/${filename}.mp3

    echo
    echo "Upload: ${basename}"

    echo
    echo " > Preprocessing..."
    sh preprocess.sh ${file} ${tempfile}

    echo
    echo " > Predicting..."
    predict=`python3 predict.py ${tempfile} | grep -o 'Predict -.*'`

    echo
    echo
    echo " > ${predict}"
    echo
    echo

    store=${predictions}/${filename}_${predict}.mp3
    sox ${file} "${store}"
    rm ${file} ${tempfile}

  done
  sleep 1
done
