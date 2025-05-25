package com.example.tikki;

import java.util.List;
import java.util.Random;
public class RandomPlayer extends Player {

    private Random random;

    /**
     * Constructor for RandomPlayer.
     * @param name The name of the player.
     */
    public RandomPlayer(String name) {
        super(name, null); // Pass null for brains since no AI model is used
        this.random = new Random(); // Initialize random number generator
    }

    /**
     * Overrides playBet to return a random bet.
     * @param tikki The game state object.
     * @return A randomly selected bet: "pass", "win", or "2-win".
     */
    @Override
    public String playBet(Tikki tikki) {
        String[] bets = {"pass", "win", "2-win"};
        Random random = new Random();

        int rand = random.nextInt(100);
        if (rand < 70) {
            return bets[0]; // pass (70%)
        } else if (rand < 99) {
            return bets[1]; // win (29%)
        } else {
            return bets[2]; // 2-win (1%)
        }
    }

    /**
     * Overrides playCard to play a random valid card from the hand.
     * @param tikki The game state object.
     * @return The randomly selected card played.
     * @throws IllegalStateException If no valid cards are available.
     */
    @Override
    public Cards playCard(Tikki tikki) {
        List<Cards> valid = validCards(tikki); // Get list of valid cards
        if (valid.isEmpty()) {
            throw new IllegalStateException("No valid cards to play");
        }
        int index = random.nextInt(valid.size()); // Randomly select an index
        Cards selected = valid.get(index); // Get the selected card
        hand.remove(selected); // Remove it from the hand
        return selected; // Return the played card
    }
}