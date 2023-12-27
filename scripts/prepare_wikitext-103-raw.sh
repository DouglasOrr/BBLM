OUT=data/wikitext-103-raw
TMPZIP=wikitext-103.zip

set -e
set -o xtrace

wget "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip" -O "${TMPZIP}"
mkdir -p "${OUT}"
unzip -j "${TMPZIP}" -d "${OUT}"
sed -i 's/^ //g' ${OUT}/*.raw
sed -i 's/ $//g' ${OUT}/*.raw
mv "${OUT}/wiki.train.raw" "${OUT}/train.txt"
mv "${OUT}/wiki.valid.raw" "${OUT}/valid.txt"
mv "${OUT}/wiki.test.raw" "${OUT}/test.txt"
rm "${TMPZIP}"
