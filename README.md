Video2Dyn
=========

- To convert the .csv output to Jules' code 
`python src/julesConverter.py --input path/data.csv --output path/output.csv --fps 30.0`

Default fps = 30

- Usage: 
    `python sequence_fit.py <dir> --orig-res 6020x4024 --out states.csv` 
It writes a CSV (smoothed and raw states, per-frame residual) plus a QA plot, 
sequence_qa_demo.png for the demo.

- `python animate_csv.py states.csv --wheelbase 1.11 --wheel-diameter 0.6604 --cam-height 0.45 --hfov-deg 45.5 --out bike.mp4`

To run the viewer
`python src/model_viewer.py`

To run the viewer and a specific data frame
`python src/model_viewer.py --label /path/to/data.txt`

To run all the tests
`pytest tests/ -v`

To run a specific set of tests
`pytest -q tests/test_bike_model.py`

To run the visual test for projection
`python -m tests.test_bike_model`

Example usage for filtered data:
`python ellipse2data.py /path/to/folder/ --cutoff 1.5 --fs 30`

Example for raw data (and signed differences):
`python ellipse2data.py /path/to/folder/ --raw --signed`

Example for filtered data with csv output:
`python ellipse2data.py /path/to/folder --output filename.csv --cutoff 1.5 --fs 30`


Example usage for the big code for single frame and plot the model:
`python src/frame_data.py -sf 1 -p mbd`

Example for data ellipse animation:
`python src/frame_data.py -d cut -a3d`

`process_imu.py` works with sensorbike conda environment




Directory structure
-------------------

```
vid2dyn/
|   README.md
|   vid2dyn-env.yml
|   LICENSE
|   AI_disclaimer.md
|
|___src/
|   |   bike_model_perspective.py
|   |   bike_model.py
|   |   ellipse2data3.py
|   |   main.py
|   |   model_perspective.py
|   |   utils.py
|
|___output/
|   |   data4model.json
|   |___animation/
|   |___plot/
|
|___archive/
|   |   kine_fit.ipynb
|   |   single_state.ipynb
|   |   video_states.ipynb
|
|___tests/
|   |   test_bike_model.py
|   |   
|___archive/
|   |   code_try.py
|   |   codetry.ipynb
|   |   model_tests.py
|   |   projection_try.ipynb
|   |   projection_try.py
|   |   test_code.ipynb
```

