#!/bin/bash

# Created on 20 July 2017
# @author: pow-pow

# If no audio file specified, or empty string
if [[ $# -lt 2 ]] || [[ -z $1 ]] || [[ -z $2 ]] ; then
  echo 'Arguments: <audio input dir> <audio output name>'
  exit 1
fi

input=$1
output=$2

dirname=`dirname ${output}`
filename=`basename ${output}`

# Transformations
convert=${dirname}/convert_${filename}
sample=${dirname}/sample_${filename}
normal=${dirname}/normal_${filename}
profile=${dirname}/${filename}.profile
reduce=${dirname}/reduce_${filename}
silence=${dirname}/silence_${filename}
reshape=${dirname}/reshape_${filename}

# Convert to MP3
sox ${input} ${convert}

# Adjust sample rate and convert to mp3
sox ${convert} -r 44100 ${sample}

# Normalize
sox --norm ${sample} ${normal}

# Noise reduction
sox ${normal} -n trim 1.5 0.4 noiseprof ${profile}
sox ${normal} ${reduce} noisered ${profile} 0.4

# Trim long period of silence at start
sox ${reduce} ${silence} silence 1 0.1 1% -1 0.1 1%

# Pad or trim audio to 2 seconds
duration=`sox --i -D ${silence}`
needspadding=`echo "${duration} < 2" | bc`

if [[ ${needspadding} -eq 1 ]] ; then

  padding=`echo "2 - ${duration}" | bc`
  sox ${silence} ${reshape} pad 0 ${padding}

else

  trimming=`echo "${duration} - 2" | bc`
  sox ${silence} ${reshape} trim 0 2

fi

# Output
sox ${reshape} ${output}

# Remove temporary files
rm ${convert} ${sample} ${normal} ${profile} ${reduce} ${silence} ${reshape}

echo "${output}"
