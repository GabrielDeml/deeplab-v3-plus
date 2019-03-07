#!/bin/bash
rm -rf captmp
mkdir -p captmp 
mkdir -p out
for video in ./*.avi
do ffmpeg -i "$video" -vf fps=1/2 captmp/"$video"-%04d.jpg
done 
for f in captmp/*.jpg 
do echo -n . 
jpegtran -rotate 180 -outfile captmp/temp.jpg "$f" 
mv captmp/temp.jpg "$f" 
done
cp captmp/* out/
rm -rf captmp

