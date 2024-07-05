from fastapi import FastAPI, HTTPException
from sqlmodel import SQLModel, Field, Session, create_engine
from typing import Optional

app = FastAPI()

# Define your SQLModel
class ChessPlayer(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    rating: int

# Database connection
DATABASE_URL = "postgresql://knut:dengamletraver@localhost/mydb"
engine = create_engine(DATABASE_URL, echo=True)

# Create tables
SQLModel.metadata.create_all(engine)

@app.post("/players/")
def create_player(player: ChessPlayer):
    with Session(engine) as session:
        session.add(player)
        session.commit()
        session.refresh(player)
        return player

@app.get("/players/{player_id}")
def read_player(player_id: int):
    with Session(engine) as session:
        player = session.get(ChessPlayer, player_id)
        if player is not None:
            return player
        raise HTTPException(status_code=404, detail="Player not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)