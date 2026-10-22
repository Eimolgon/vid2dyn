Video2Data
==========


- Usage: python sequence_fit.py <dir> --orig-res 6020x4024 --out states.csv. 
It writes a CSV (smoothed and raw states, per-frame residual) plus a QA plot, 
sequence_qa_demo.png for the demo.

- python animate_csv.py states.csv --wheelbase 1.11 --wheel-diameter 0.6604 
--cam-height 0.45 --hfov-deg 45.5 --out bike.mp4

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


To-Do
-------

- [ ] Test reconstruction accuracy
- [x] Argparse
    - [ ] Improve argparse plot: -p all, -p yaw, -p roll, etc.
    - [ ] Add -t for tests -t model -t ellipse, etc.
- [ ] Fix filtered data processing
- [ ] Use previous frames to estimate next state



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


Dataset - camera motion
-----------------------

<details> <summary> Camera motion </summary>
    
1. Moving
2. Static
3. Static
4. Moving
5. Static
6. Static
7. Moving
8. Static
9. Static
10. Static
11. Static
12. Moving
13. Moving
14. Moving
15. Moving
16. Static
17. Moving
18. Moving
19. Static
20. Moving
21. Moving
22. Moving
23. Moving
24. Moving
25. Moving
26. Static
27. Moving
28. Moving
29. Moving
30. Static
31. Moving
32. Static
33. Static
34. Static
35. Static
36. Static
37. Static
38. Static
39. Static
40. Static
41. Static
42. Static
43. Static
44. Moving
45. Moving
46. Moving
47. Moving
48. Moving
49. Moving
50. Static
51. Static
52. Static
53. Moving
54. Moving
55. Moving
56. Moving
57. Static
58. Moving
59. Moving
60. Moving
61. Static
62. Moving
63. Moving
64. Static
65. Moving
66. Moving
67. Moving
68. Moving
69. Moving
70. Static
71. Moving
72. Moving
73. Moving
74. Moving
75. Moving
76. Static
77. Moving
78. Moving
79. Moving
80. Static
81. Static
82. Moving
83. Moving
84. Moving
85. Moving
86. Moving
87. Moving
88. Static
89. Moving
90. Static
91. Moving
92. Static
93. Static
94. Moving
95. Moving
96. Moving
97. Static
98. Static
99. Moving
100. Moving

</details>

Notes
-----

-  argparse options: single frame, all frames, plot, animate.
    - Default: all frames, no animation, no plot, print results.

- how to test reconstruction accuracy?
