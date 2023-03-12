root_dir=/data/MICCAI23_samples/2SCMv6/counterfactual
video_dir=$root_dir/videos
image_real=$root_dir/images/real
image_fake=$root_dir/images/fake

i=0
for file in $video_dir/*
do
    # get base name of file
    name=$(basename "$file")
    # remove extension
    name="${name%.*}"
    echo $name

    ###############################
    # For REAL images (left crop) #
    ###############################
    # create directory
    mkdir -p $image_real/$name

    # transform videos into frames + pad to 128x128
    convert $file -crop 112x112+0+0 -background black -gravity center -extent 128x128 +repage $image_real/$name/%d.jpg

    ###############################
    # For FAKE images (right crop) #
    ###############################
    # create directory
    mkdir -p $image_fake/$name

    # transform videos into frames + pad to 128x128
    convert $file -crop 112x112+112+0 -background black -gravity center -extent 128x128 +repage $image_fake/$name/%d.jpg


    # if [ $i -eq 2 ]
    # then
    #     break
    # fi
    i=$((i+1))
done
echo $i

# compute metrics with:
# python src/scripts/calc_metrics_for_dataset.py --real_data_path /path/to/dataset_a.zip --fake_data_path /path/to/dataset_b.zip --mirror 0 --gpus 1 --resolution 128 --metrics fvd2048_16f,fid50k_full
