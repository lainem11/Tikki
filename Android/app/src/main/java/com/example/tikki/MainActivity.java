package com.example.tikki;

import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.content.ContextCompat;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;

import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.IValue;
//import org.pytorch.LiteModuleLoader;
import org.w3c.dom.Text;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.util.Log;
import android.view.View;
import android.widget.Toast;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;

import java.util.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;



public class MainActivity extends AppCompatActivity {

    private Button tikkiButton, kuolemaButton, chichiButton;
    private EditText playerName;
    private String enemyName;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tikkiButton = findViewById(R.id.tikki_button);
        tikkiButton.setEnabled(false);
        kuolemaButton = findViewById(R.id.button_kuolema);
        chichiButton = findViewById(R.id.button_chichi);
        EditText textBar = findViewById(R.id.text_player_name);

        tikkiButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, GameActivity.class);
                String enteredText = textBar.getText().toString();
                if (enteredText.equals("")) {
                    enteredText = "P1";
                }
                intent.putExtra("playerName", enteredText);
                intent.putExtra("enemyModel",enemyName);
                startActivity(intent);
            }
        });

        kuolemaButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                kuolemaButton.setBackgroundColor(ContextCompat.getColor(MainActivity.this, R.color.button_pressed));
                chichiButton.setBackgroundColor(ContextCompat.getColor(MainActivity.this, R.color.button_default));
                tikkiButton.setEnabled(true);
                enemyName = "mobile_model6.pt";
            }
        });

        chichiButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                kuolemaButton.setBackgroundColor(ContextCompat.getColor(MainActivity.this, R.color.button_default));
                chichiButton.setBackgroundColor(ContextCompat.getColor(MainActivity.this, R.color.button_pressed));
                tikkiButton.setEnabled(true);
                enemyName = "mobile_model4.pt";
            }
        });
    }

}