import logging
import time
from http.client import HTTPException
from fastapi import BackgroundTasks
from pydantic import BaseModel
# from fastapi.responses import HTMLResponse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()
origins = [
    '*',  # Allow all origins
    "http://localhost:3000",
    "localhost:3000",
    'https://chess-state-front.vercel.app/'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
cur_data = {}


@app.post('/post_data/{data_type}')
async def post_data(data_type: str, rows: List[List[str]]) -> dict:
    cur_data = parse_json_rows(data_type, rows)


async def _private_method(fen, from_square, mode, to_square, username, background_tasks):
    start = time.time()
    if username not in valid_usernames:
        raise HTTPException(status_code=400, detail='Invalid username')
    print('Getting Backend', time.time()-start)
    backend = backends[username]
    fen = Fen.decode(fen)
    try:
        print('Pushing to backend', time.time()-start)
        board, feedback = backend.board_push(fen, chess.Move.from_uci(from_square + to_square), mode)
    except RedisError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not feedback.is_correct:
        mode = 'show'
    else:
        mode = {'show': 'repeat',
                'repeat': 'play',
                'play': 'play'}[mode]
    print('Fetching score', time.time()-start)
    white_score, black_score = backend.fetch_score()
    response = {"board": board.fen(),
                'is_correct': feedback.is_correct,
                'correct_move': feedback.correct_move,
                'mode': mode,
                'white_score': white_score,
                'black_score': black_score
                }
    print(f'Prepared response {response}', time.time()-start)
    print('Adding tasks', time.time()-start)
    background_tasks.add_task(backend.load_if_not)
    background_tasks.add_task(backend.set_score)
    states[username] = State(board.fen())
    print(f'{time.time() - start:.2f} seconds')
    return response


def main_backend():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
