package com.example.tikki;

import java.util.Objects;

public class Cards {
    private String suit;
    private String rank;

    public Cards(String suit, String rank) {
        this.suit = suit;
        this.rank = rank;
    }

    public String getSuit() {
        return this.suit;
    }

    public String getRank() {
        return this.rank;
    }

    @Override
    public String toString() {
        return suit + rank;
    }

    @Override
    public int hashCode() {
        return Objects.hash(suit, rank);
    }
}
