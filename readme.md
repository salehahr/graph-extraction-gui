# graph-extraction-gui
A GUI to visualise an NN-based
[graph extraction procedure](http://github.com/salehahr/graph-extraction-networks)
applied to skeletonised endoscopic images.

![](./assets/gui.png)

## Notes
- Sample skeletonised images are provided in the [`img`](./img) folder.
- The adjacency matrix is predicted using the BAC scheme.
    The number of neighbours `k0` can be adjusted in [`config.py`](./config.py)
- This GUI implementation is currently not optimised for GPU.

## Running using a remote interpreter
- Set the following environment variables (example)
    ```
    DISPLAY=localhost:10;LIBGL_ALWAYS_INDIRECT=1
    ```
- Set the corresponding display number in the GUI viewer (e.g. VcXsrv)
