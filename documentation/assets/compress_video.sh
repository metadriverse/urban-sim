# #!/bin/bash

# INPUT_DIR="$1"
# OUTPUT_DIR="$2"

# if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
#     echo "Usage: $0 <input_folder> <output_folder>"
#     exit 1
# fi

# mkdir -p "$OUTPUT_DIR"

# declare -a FILENAMES
# declare -a ORIGINAL_SIZES
# declare -a COMPRESSED_SIZES

# for file in "$INPUT_DIR"/*.mp4; do
#     [ -e "$file" ] || continue
#     filename=$(basename "$file")
#     orig_size=$(du -m "$file" | cut -f1)
#     FILENAMES+=("$filename")
#     ORIGINAL_SIZES+=("$orig_size")

#     if [ "$INPUT_DIR" = "$OUTPUT_DIR" ]; then
#         temp_file="$INPUT_DIR/.tmp_$filename"
#         ffmpeg -i "$file" -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k -movflags +faststart -y "$temp_file" < /dev/null
#         comp_size=$(du -m "$temp_file" | cut -f1)
#         mv "$temp_file" "$file"
#     else
#         output_file="$OUTPUT_DIR/$filename"
#         ffmpeg -i "$file" -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k -movflags +faststart -y "$output_file" < /dev/null
#         comp_size=$(du -m "$output_file" | cut -f1)
#     fi
#     COMPRESSED_SIZES+=("$comp_size")
# done

# echo
# echo "Compression Summary:"
# echo

# for i in "${!FILENAMES[@]}"; do
#     orig=${ORIGINAL_SIZES[$i]}
#     comp=${COMPRESSED_SIZES[$i]}
#     percent=0
#     if [ "$orig" -gt 0 ]; then
#         percent=$(( 100 - (100 * comp / orig) ))
#     fi
#     printf "%s: %sMB → %sMB (%d%% reduction)\n" "${FILENAMES[$i]}" "$orig" "$comp" "$percent"
# done
#!/bin/bash

INPUT_DIR="$1"
OUTPUT_DIR="$2"

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

declare -a FILENAMES

for file in "$INPUT_DIR"/*.mp4; do
    [ -e "$file" ] || continue
    filename=$(basename "$file" .mp4)
    FILENAMES+=("$filename")

    output_file="$OUTPUT_DIR/$filename.gif"

    # Convert first 2 seconds to GIF at 10fps and resize to 640 width
    ffmpeg -t 2 -i "$file" -vf "fps=10,scale=640:-1:flags=lanczos" -y "$output_file" < /dev/null
done

echo
echo "Conversion to GIF completed for:"
for name in "${FILENAMES[@]}"; do
    echo "- $name.mp4 → $name.gif"
done
