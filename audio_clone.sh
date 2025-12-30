# cd /home/csgrad/jsun39/OpenVoice

python3 audio_clone.py \
  --input_dir /home/csgrad/jsun39/OpenVoice/dataset/english_free_speech/files_cut_by_sentences \
  --output_dir /home/csgrad/jsun39/OpenVoice/1_different_stytle \
  --base_dir checkpoints/base_speakers/EN \
  --converter_dir checkpoints/converter \
  --styles default,friendly,cheerful,excited,sad,angry,terrified,shouting,whispering \
  --speed 1.0 \
  --limit 2