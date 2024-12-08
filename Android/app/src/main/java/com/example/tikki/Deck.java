package com.example.tikki;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Deck {
    private List<Cards> cards;

    public Deck() {
        String[] suits = {"♥", "♦", "♣", "♠"};
        String[] ranks = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"};
        cards = new ArrayList<>();

        for (String suit : suits) {
            for (String rank : ranks) {
                cards.add(new Cards(suit, rank));
            }
        }
    }

    public List<Cards> getCards() {
        return cards;
    }

    public void shuffle(long seed) {
        Random random = new Random(seed);
        for (int i = cards.size() - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            Cards temp = cards.get(i);
            cards.set(i, cards.get(j));
            cards.set(j, temp);
        }
    }

    public Cards draw() {
        if (cards.isEmpty()) {
            return null;
        }
        return cards.remove(cards.size() - 1);
    }
}