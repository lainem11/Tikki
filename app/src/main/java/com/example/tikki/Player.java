package com.example.tikki;

import android.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.IValue;
//import org.pytorch.LiteModuleLoader;



public class Player {
    public String name;
    private List<Cards> hand;
    private Module brains = null;
    private Deck deck;
    private int n_players = 2;
    private int round_n = 5;
    private int card_n = 5;
    private int n_bets = 3;
    private int n_inputs = card_n + n_players*(1+round_n) + n_players;
    private int n_outputs = 52+n_bets;
    private Map<Integer, Integer> cards;

    public Player(String name, Module brains) {
        this.name = name;
        this.hand = new ArrayList<>();
        this.brains = brains;
        this.deck = new Deck();

        // Maps cards to indices
        cards = new HashMap<>();
        List<Cards> deckCards = deck.getCards();
        for (int i = 0; i < deckCards.size(); i++) {
            cards.put(deckCards.get(i).hashCode(), i + 1);
        }
    }

    public List<Cards> getHand() {
        return this.hand;
    }

    public void setHand(ArrayList<Cards> newHand) {
        this.hand = newHand;
    }
    public void drawCard(Deck deck, int n) {
        for (int i = 0; i < n; i++) {
            Cards card = deck.draw();
            if (card != null) {
                hand.add(card);
            } else {
                return;
            }
            if (hand.size() > 5) {
                throw new IllegalArgumentException("Illegal draw!");
            }
        }
    }

    public int[] getState(Tikki tikki) {
        // Create an array for my cards (5 cards max)
        int[] iCards = new int[5];
        List<Cards> hand = this.getHand();
        for (int i = 0; i < hand.size(); i++) {
            iCards[i] = cards.get(hand.get(i).hashCode());
        }

        int nPlayers = tikki.players.size();
        int histSize = nPlayers * (1 + 5);
        int[] iHist = new int[histSize];
        int inputCounter = 0;

        HashMap<Integer, List<HashMap<Player, Object>>> playHistory = tikki.playHistory;

        // Populate play history
        for (int r = 0; r < playHistory.size(); r++) {
            List<HashMap<Player, Object>> roundHistory = playHistory.get(r);
            for (HashMap<Player, Object> moveEntry : roundHistory) {
                for (Map.Entry<Player, Object> entry : moveEntry.entrySet()) {
                    Player player = entry.getKey();
                    Object move = entry.getValue();

                    if (r != 0) {
                        if (move instanceof Cards) {
                            iHist[inputCounter] = cards.get(((Cards) move).hashCode());
                        }
                    } else {
                        int bet;
                        switch (move.toString()) {
                            case "pass":
                                bet = 53;
                                break;
                            case "win":
                                bet = 54;
                                break;
                            case "2-win":
                                bet = 55;
                                break;
                            default:
                                bet = 0;
                                break;
                        }
                        iHist[inputCounter] = bet;
                    }
                    inputCounter++;
                }
            }
        }

        // Score vocabulary
        Map<Integer, Integer> scoreVocab = new HashMap<>();
        for (int i = -10, index = 56; i <= 5; i++, index++) {
            scoreVocab.put(i, index);
        }

        // Scores for all players
        int[] iScore = new int[nPlayers];
        Map<Player, Integer> gameScores = tikki.playerScores;

        List<Player> tikkiPlayers = new ArrayList<>(tikki.players);
        // Rotate players so that "this" player is first
        Collections.rotate(tikkiPlayers, -tikkiPlayers.indexOf(this));

        for (int ind = 0; ind < tikkiPlayers.size(); ind++) {
            Player player = tikkiPlayers.get(ind);
            iScore[ind] = scoreVocab.get(gameScores.get(player));
        }

        // Combine iCards, iHist, and iScore into a single array
        int nInputs = iCards.length + iHist.length + iScore.length;
        int[] state = new int[nInputs];
        System.arraycopy(iCards, 0, state, 0, iCards.length);
        System.arraycopy(iHist, 0, state, iCards.length, iHist.length);
        System.arraycopy(iScore, 0, state, iCards.length + iHist.length, iScore.length);

        return state;
    }

    private float[] chooseAction(Tikki tikki) {
        Log.d("Choosing action", "Generating game state");
        int [] int_state = this.getState(tikki);
        long[] long_state = Arrays.stream(int_state).mapToLong(i -> i).toArray();
        Tensor state = Tensor.fromBlob(long_state,new long[]{1,n_inputs});
        Log.d("Action", "Passing state to net");
        Tensor output = brains.forward(IValue.from(state)).toTensor();
        return output.getDataAsFloatArray();
    }

    public String playBet(Tikki tikki) {
        String index;
        if (brains == null) {
            index = "pass";
        } else {
            float [] action_array = chooseAction(tikki);
            int startIndex = n_outputs - 3;
            int endIndex = n_outputs - 1;
            int maxIndex = startIndex;
            for (int i = startIndex; i <= endIndex; i++) {
                if (action_array[i] > action_array[maxIndex]) {
                    maxIndex = i;
                }
            }
            switch (maxIndex) {
                case 52:
                    index = "pass";
                    break;
                case 53:
                    index = "win";
                    break;
                case 54:
                    index = "2-win";
                    break;
                default:
                    index = "2-win";
                    break;
            }
        }
        return index;
    }

    public Cards playCard(Tikki tikki) {
        List<Cards> validCards = validCards(tikki);

        if (brains == null) {
            while (true) {
                int index = new Random().nextInt(5);
                if (index < hand.size() && validCards.contains(hand.get(index))) {
                    return hand.remove(index);
                }
            }
        } else {
            int maxIndex = -1;
            Cards bestCard = null;
            float maxValue = -Float.MAX_VALUE;
            float [] cardValues = chooseAction(tikki);
            for (int i = 0; i < hand.size(); i++) {
                Cards card = hand.get(i);
                if (validCards.contains(card)) {
                    float cardValue = cardValues[cards.get(card.hashCode())];

                    if (cardValue > maxValue) {
                        maxValue = cardValue;
                        bestCard = card;
                        maxIndex = i;
                    }
                }
            }
            Log.d("Looped thorugh valid cards", "About to remove card from hand");
            if (maxIndex != -1) {
                return hand.remove(maxIndex);
            }

            throw new IllegalStateException("No valid cards to play");
        }

    }

    public Cards playCard(Tikki tikki, Cards selectedCard) {
        if (validCards(tikki).contains(selectedCard)) {
            int selectedCardIndex = hand.indexOf(selectedCard);
            return hand.remove(selectedCardIndex);
        } else {
            Log.d("Card play failed", "Invalid card selected to play");
            return null;
        }
    }

    public void showHand() {
        System.out.println(name + "'s hand:");
        StringBuilder str = new StringBuilder();
        for (Cards card : hand) {
            str.append(" ").append(card.getSuit()).append(card.getRank());
        }
        System.out.println(str);
    }

    public List<Cards> validCards(Tikki tikki) {
        boolean forcedFlag = hand.stream().anyMatch(card -> card.getSuit().equals(tikki.getRoundSuit()));
        if (!forcedFlag) {
            return hand;
        }
        List<Cards> validCards = new ArrayList<>();
        for (Cards card : hand) {
            if (card.getSuit().equals(tikki.getRoundSuit())) {
                validCards.add(card);
            }
        }
        return validCards;
    }
}
