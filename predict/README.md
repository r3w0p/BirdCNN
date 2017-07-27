# Predict

## Edison Demonstration
To demonstrate preprocessing and prediction upon sample data, run:
- `sh runEdisonHandler.sh`

To view files in the browser at `localhost:8000`, run:
- `sh runEdisonServer.sh`

The `edison/uploads` directory comes with audio files already: `sound-12-10-01` and `sound-12-10-02` are Barn Owl calls and the other three are Crow caws.

## Another Example
Using the provided sample `crow_example.wav`, run:
- `sh preprocess.sh crow_example.wav crow.mp3`
- `python3 predict.py crow.mp3`

This should produce a correct prediction of `Crow` with a probability of `97.4%`.
