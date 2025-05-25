package com.example.tikki;

import java.util.*;

public class Tikki {
    public List<Player> players;
    private Player humanPlayer;
    public int maxRounds; // in practice +1 because of betting round
    private int maxScore;
    private boolean matchStarted;
    private String roundSuit;
    public int roundNumber;
    public HashMap<Integer, List<HashMap<Player, Object>>> playHistory;
    public Map<Player, Integer> playerScores;

    public int currentPlayerIndex;
    public int turnCount;
    private Deck deck;
    private Player matchStarter;
    private Player roundWinner;
    public Player gameWinner;
    public boolean gameOver = false;

    public Tikki(List<Player> players) {
        this.players = players;
        this.maxRounds = 5;
        this.maxScore = 5;
        this.resetScores();
        this.humanPlayer = this.players.get(0);
    }

    public void resetScores() {
        if (playerScores != null) {
            playerScores.replaceAll((player, score) -> 0);
        } else {
            // Optional: Initialize if needed or log a warning
            playerScores = new HashMap<>();
        }
    }

    public String getRoundSuit() {
        return this.roundSuit;
    }
    public int playGame() {
        newGame();
        boolean complete = false;
        while (!complete) {
            // Assuming nextTurn method exists
            // Replace the following line with actual implementation
            // For now, just sleep for 1 millisecond
            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            // Assuming done variable is set correctly
            // Replace the following line with actual implementation
            // For now, just set done to true
            boolean done = true;
            if (done) {
                complete = true;
            }
        }
        return 1;
    }
    // Convert the scores map to a readable string
    public String getScoreString() {
        StringBuilder scoreBuilder = new StringBuilder();
        for (Map.Entry<Player, Integer> entry : playerScores.entrySet()) {
            scoreBuilder.append(entry.getKey().name)
                    .append(": ")
                    .append(entry.getValue())
                    .append(", ");
        }
        // Remove the last comma and space if the map isn't empty
        if (scoreBuilder.length() > 0) {
            scoreBuilder.setLength(scoreBuilder.length() - 2);
        }
        return scoreBuilder.toString();
    }
    public void newGame() {
        // Initialize game state
        this.roundWinner = null;
        this.gameWinner = null;
        this.playerScores = new HashMap<Player,Integer>();
        for (Player p : players) {
            playerScores.put(p, 0);
        }
        this.currentPlayerIndex = new Random().nextInt(players.size());
        this.matchStarter = players.get(currentPlayerIndex);
        this.turnCount = 0;
        this.roundSuit = "";
        this.playHistory = new HashMap<Integer, List<HashMap<Player, Object>>>();
        this.matchStarted = false;
        for (Player p : players) {
            p.setHand(new ArrayList<>());
        }
        newMatch();
    }

    private void newMatch() {
        this.matchStarted = true;

        this.roundNumber = 0;
        this.roundSuit = "";
        this.playHistory = new HashMap<Integer, List<HashMap<Player, Object>>>();
        for (int i = 0; i < this.maxRounds+1 ; i++) {
            this.playHistory.put(i, new ArrayList<>());
        }
        this.deck = new Deck();
        this.deck.shuffle(new Random().nextInt());
        for (Player p : players) {
            p.drawCard(this.deck, this.maxRounds);
        }
    }

    public void endGame() {
        this.gameOver = true;
    }
    public Object[] nextTurn(Object playedMove) {
        int done = 0;
        int terminated = 0;
        int played_card_index = 0;
        String moveString = "";
        Player currentPlayer = players.get(this.currentPlayerIndex);
        // Automatically start new match if needed
        if (!this.matchStarted) {
            this.newMatch();
            this.matchStarter = currentPlayer;
        }
        if (this.roundNumber == 0) {
            if (playedMove == null) {
                moveString = currentPlayer.playBet(this);
            } else {
                moveString = playedMove.toString();
            }
            HashMap<Player, Object> playerMoveMap = new HashMap<>();
            playerMoveMap.put(currentPlayer, moveString);
            this.playHistory.get(this.roundNumber).add(playerMoveMap);
            if (this.turnCount == (players.size() - 1)) {
                this.roundNumber += 1;
            }
            this.currentPlayerIndex = (this.currentPlayerIndex + 1) % players.size();
        } else {
            Object[] play_feedback;
            Cards moveCard = null;
            if (playedMove == null) {
                moveCard = (Cards) currentPlayer.playCard(this);
            } else {
                moveCard = (Cards) playedMove;
                currentPlayer.playCard(this,moveCard);
            }

            //played_card_index = (int) play_feedback[1];
            moveString = moveCard.getSuit() + moveCard.getRank();
            HashMap<Player, Object> playerMoveMap = new HashMap<>();
            playerMoveMap.put(currentPlayer, moveCard);
            this.playHistory.get(this.roundNumber).add(playerMoveMap);
            if (this.turnCount == (players.size() - 1)) {
                int maxRank = 0;
                List<HashMap<Player, Object>> roundHistory = this.playHistory.get(this.roundNumber);
                for (HashMap<Player, Object> playerData : roundHistory) {
                    for (HashMap.Entry<Player, Object> entry : playerData.entrySet()) {
                        Player player = entry.getKey();
                        Object object = entry.getValue();
                        Cards card = (Cards) object;
                        if (card.getSuit().equals(this.roundSuit) && (Integer.parseInt(card.getRank()) > maxRank)) {
                            maxRank = Integer.parseInt(card.getRank());
                            this.roundWinner = player;
                        }
                    }
                }
                if (this.roundNumber == this.maxRounds) {
                    terminated = 1;
                    this.matchStarted = false;
                    giveScore();
                    if (this.playerScores.get(this.roundWinner) >= maxScore) {
                        done = 1;
                        this.gameWinner = this.roundWinner;
                        this.endGame();
                    } else {
                        this.currentPlayerIndex = (players.indexOf(this.matchStarter) + 1) % players.size();
                        //this.roundNumber = 0;
                        this.newMatch();
                        this.matchStarter = players.get(this.currentPlayerIndex);;
                    }
                } else {
                    this.currentPlayerIndex = players.indexOf(this.roundWinner);
                    this.roundSuit = "";
                    this.roundNumber += 1;
                }
            } else {
                this.currentPlayerIndex = (this.currentPlayerIndex + 1) % players.size();
                if (this.turnCount == 0) {
                    this.roundSuit = moveCard.getSuit();
                }
            }
        }
        this.turnCount = (this.turnCount + 1) % players.size();
        return new Object[]{moveString, played_card_index, terminated, done};
    }

    public void giveScore() {
        List<HashMap<Player, Object>> firstRoundHistory = this.playHistory.get(0);
        for (HashMap<Player, Object> firstRoundPlayerData : firstRoundHistory) {
            for (Map.Entry<Player, Object> entryFirstRound : firstRoundPlayerData.entrySet()) {
                Player playerFirstRound = entryFirstRound.getKey();
                Object object = entryFirstRound.getValue();
                String bet = (String) object;

                List<HashMap<Player, Object>> lastRoundHistory = this.playHistory.get(this.roundNumber);
                for (HashMap<Player, Object> lastRoundPlayerData : lastRoundHistory) {
                    for (Map.Entry<Player, Object> entryLastRound : lastRoundPlayerData.entrySet()) {
                        Player playerLastRound = entryLastRound.getKey();
                        Object last_object = entryLastRound.getValue();
                        Cards cardLastRound = (Cards) last_object;
                        if (playerFirstRound.equals(playerLastRound)) {
                            String lastCardRank = cardLastRound.getRank();
                            boolean is_winner = this.roundWinner.equals(playerFirstRound);
                            boolean won_with_2 = lastCardRank.equals("2");
                            if (is_winner) {
                                this.playerScores.put(playerFirstRound, this.playerScores.get(playerFirstRound) + 1);
                                if (won_with_2) {
                                    this.playerScores.put(playerFirstRound, this.playerScores.get(playerFirstRound) + 1);
                                }
                            }
                            switch (bet) {
                                case "win":
                                    if (is_winner) {
                                        this.playerScores.put(playerFirstRound, this.playerScores.get(playerFirstRound) + 1);
                                    } else {
                                        this.playerScores.put(playerFirstRound, this.playerScores.get(playerFirstRound) - 1);
                                    }
                                    break;
                                case "2-win":
                                    if (is_winner && lastCardRank.equals("2")) {
                                        this.playerScores.put(playerFirstRound, this.playerScores.get(playerFirstRound) + 2);
                                    } else {
                                        this.playerScores.put(playerFirstRound, this.playerScores.get(playerFirstRound) - 2);
                                    }
                                    break;
                                case "pass":
                                    break;
                                default:
                                    System.out.println("Error in bet");
                            }
                            if (this.playerScores.get(playerFirstRound) < -5) {
                                this.playerScores.put(playerFirstRound, -5);
                            }
                            if (this.playerScores.get(playerFirstRound) > 5) {
                                this.playerScores.put(playerFirstRound, 5);
                            }
                        }
                    }
                }
            }
        }
    }
}
