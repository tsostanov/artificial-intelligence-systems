:- discontiguous ability_of/2.
:- discontiguous in_hand/2.
:- discontiguous in_deck/2.
:- discontiguous in_board/2.


initial_state(game_state(
    player(30, 3, 3,
           [ card(wisp,0,1,1), card(murloc_raider,1,2,1), card(chillwind_yeti,4,4,5) ],
           [ card(stonetusk_boar,1,1,1), card(bloodfen_raptor,2,3,2), card(argent_squire,1,1,1) ],
           [ minion(ally_1,2,2,false) ]),   % ally_1 не может атаковать сразу
    player(30, 2, 2,
           [ card(river_crocolisk,2,2,3), card(boulderfist_ogre,6,6,7) ],
           [ card(scarlet_crusader,3,3,1) ],
           [ minion(enemy_1,1,1,false) ]), % enemy_1 не может атаковать сразу
    1, 1
)).

% -----------------------
% КАРТЫ (card/4: Name, Cost, Attack, Health)
% -----------------------
card(wisp, 0, 1, 1).
card(stonetusk_boar, 1, 1, 1).
card(murloc_raider, 1, 2, 1).
card(bloodfen_raptor, 2, 3, 2).
card(river_crocolisk, 2, 2, 3).
card(chillwind_yeti, 4, 4, 5).
card(boulderfist_ogre, 6, 6, 7).
card(argent_squire, 1, 1, 1).
card(scarlet_crusader, 3, 3, 1).
card(loot_hoarder, 2, 2, 1).
card(war_golem, 7, 7, 7).
card(acolyte_of_pain, 3, 1, 3).
card(leeroy_jenkins, 5, 6, 2).

% -----------------------
% ОДНОАРГУМЕНТНЫЕ ФАКТЫ (не менее 20)
% -----------------------
minion_name(wisp).
minion_name(stonetusk_boar).
minion_name(murloc_raider).
minion_name(bloodfen_raptor).
minion_name(river_crocolisk).
minion_name(chillwind_yeti).
minion_name(boulderfist_ogre).
minion_name(argent_squire).
minion_name(scarlet_crusader).
minion_name(loot_hoarder).
minion_name(war_golem).
minion_name(acolyte_of_pain).
minion_name(leeroy_jenkins).

player_name(player1).
player_name(player2).

ability(charge).
ability(taunt).
ability(rush).
ability(divine_shield).
ability(battlecry).

tribe(beast).
tribe(murloc).
tribe(mech).

rarity(common).
rarity(rare).

zone(hand).
zone(deck).
zone(board).

role(hero).

% -----------------------
% ДВУХАРГУМЕНТНЫЕ ФАКТЫ
% -----------------------
card_cost(chillwind_yeti, 4).
card_attack(chillwind_yeti, 4).
card_health(chillwind_yeti, 5).

card_cost(boulderfist_ogre, 6).
card_cost(argent_squire, 1).
card_cost(scarlet_crusader, 3).

% способности у карт
ability_of(stonetusk_boar, charge).
ability_of(argent_squire, divine_shield).
ability_of(murloc_raider, rush).
ability_of(scarlet_crusader, taunt).
ability_of(leeroy_jenkins, charge). % пример: leeroy тоже считает charge-like

% трибы
tribe_of(murloc_raider, murloc).
tribe_of(river_crocolisk, beast).

% положение карт (дополнение к initial_state; использовать для правил/запросов)
in_hand(player1, stonetusk_boar).
in_hand(player1, bloodfen_raptor).
in_hand(player2, scarlet_crusader).
in_deck(player1, chillwind_yeti).
in_board(player2, enemy_1).

% -----------------------
% ВСПОМОГАТЕЛЬНЫЕ ПРЕДИКАТЫ (утилиты)
% -----------------------
remove_from_hand(Hand, Card, NewHand) :-
    select(Card, Hand, NewHand).

replace(Index, List, NewItem, NewList) :-
    length(Prefix, Index1),
    Index is Index1 + 1,
    append(Prefix, [_|Suffix], List),
    append(Prefix, [NewItem|Suffix], NewList).

% is_dead для minion(Name,Attack,Health,CanAttack)
is_dead(minion(_, _, Health, _)) :- Health =< 0.
remove_dead_minions(Board, FinalBoard) :-
    exclude(is_dead, Board, FinalBoard).

draw_card([Card|Deck], Hand, Deck, [Card|Hand]).
draw_card([], Hand, [], Hand).

% update_minion_health для 4-аргументной структуры
update_minion_health(Board, Index, NewHealth, NewBoard) :-
    nth1(Index, Board, minion(Name, Attack, _OldH, CanAttack)),
    replace(Index, Board, minion(Name, Attack, NewHealth, CanAttack), NewBoard).

% set all minions on board to CanAttack = true (used at start of player's turn)
set_minions_ready([], []).
set_minions_ready([minion(Name, Attack, Health, _)|T], [minion(Name, Attack, Health, true)|T2]) :-
    set_minions_ready(T, T2).

% -----------------------
% ПРЕДИКАТЫ ИГРОВОГО ПРОЦЕССА (c charge/rush)
% -----------------------
% play_minion(GameState, MinionName, NewGameState)
% Если карта имеет charge -> CanAttack = true.
% Если карта имеет rush -> CanAttack = true, но в атаке будет запрещён таргет 0 (герой).
% Иначе CanAttack = false (summon sickness).

play_minion(GameState, MinionName, NewGameState) :-
    GameState = game_state(P1, P2, CurrentPlayer, Turn),
    CurrentPlayer = 1,
    P1 = player(H1, M1, MaxM1, Deck1, Hand1, Board1),
    member(card(MinionName, Cost, Attack, Health), Hand1),
    Cost =< M1,
    NewMana is M1 - Cost,
    remove_from_hand(Hand1, card(MinionName, Cost, Attack, Health), NewHand1),
    % determine if the minion can attack immediately
    ( ability_of(MinionName, charge) -> CanAttack = true
    ; ability_of(MinionName, rush)   -> CanAttack = true
    ; CanAttack = false
    ),
    NewBoard1 = [minion(MinionName, Attack, Health, CanAttack)|Board1],
    NewP1 = player(H1, NewMana, MaxM1, Deck1, NewHand1, NewBoard1),
    NewGameState = game_state(NewP1, P2, CurrentPlayer, Turn).

% attack(GameState, AttackerIndex, TargetIndex, NewGameState)
% Проверки:
% - атакующий должен иметь CanAttack = true
% - если атакующий имеет rush, то TargetIndex = 0 (герой) запрещён
% После атаки у атакующего CanAttack -> false
attack(GameState, AttackerIndex, TargetIndex, NewGameState) :-
    GameState = game_state(P1, P2, CurrentPlayer, Turn),
    CurrentPlayer = 1,
    P1 = player(H1, M1, MaxM1, Deck1, Hand1, Board1),
    P2 = player(H2, M2, MaxM2, Deck2, Hand2, Board2),
    nth1(AttackerIndex, Board1, minion(AttackerName, AttackerAttack, AttackerHealth, AttackerCanAttack)),
    % проверка, что может атаковать
    ( AttackerCanAttack = true ->
        true
    ;
        % нельзя атаковать — завершаем как failure
        fail
    ),
    % если у атакующего есть rush, запрещаем атаку в героя
    ( ability_of(AttackerName, rush), TargetIndex = 0 ->
        % rush не может атаковать героя
        fail
    ;
        true
    ),
    ( TargetIndex = 0 ->
        % Атака героя
        NewH2 is H2 - AttackerAttack,
        % поставить флаг CanAttack = false для атакующего
        replace(AttackerIndex, Board1, minion(AttackerName, AttackerAttack, AttackerHealth, false), TempBoard1),
        NewP2 = player(NewH2, M2, MaxM2, Deck2, Hand2, Board2),
        NewP1 = player(H1, M1, MaxM1, Deck1, Hand1, TempBoard1),
        NewGameState = game_state(NewP1, NewP2, CurrentPlayer, Turn)
    ;
        % Атака вражеского существа
        nth1(TargetIndex, Board2, minion(TargetName, TargetAttack, TargetHealth, TargetCanAttack)),
        NewAttackerHealth is AttackerHealth - TargetAttack,
        NewTargetHealth is TargetHealth - AttackerAttack,
        % обновляем доски (и делаем атакующему CanAttack=false)
        replace(AttackerIndex, Board1, minion(AttackerName, AttackerAttack, NewAttackerHealth, false), TempBoard1),
        replace(TargetIndex, Board2, minion(TargetName, TargetAttack, NewTargetHealth, TargetCanAttack), TempBoard2),
        remove_dead_minions(TempBoard1, FinalBoard1),
        remove_dead_minions(TempBoard2, FinalBoard2),
        NewP1 = player(H1, M1, MaxM1, Deck1, Hand1, FinalBoard1),
        NewP2 = player(H2, M2, MaxM2, Deck2, Hand2, FinalBoard2),
        NewGameState = game_state(NewP1, NewP2, CurrentPlayer, Turn)
    ).

% end_turn / prepare_player_turn
% На начале хода у игрока все его миньоны готовы (CanAttack = true),
% т.е. снимается сумон-сикнесс у тех, кто был на доске в начале хода.
end_turn(GameState, NewGameState) :-
    GameState = game_state(P1, P2, CurrentPlayer, Turn),
    NextPlayer is 3 - CurrentPlayer,
    NextTurn is Turn + 1,
    ( CurrentPlayer = 1 ->
        prepare_player_turn(P2, NextTurn, NewP2),
        NewGameState = game_state(P1, NewP2, NextPlayer, NextTurn)
    ;
        prepare_player_turn(P1, NextTurn, NewP1),
        NewGameState = game_state(NewP1, P2, NextPlayer, NextTurn)
    ).

prepare_player_turn(player(H, _M, MaxM, Deck, Hand, Board), _Turn, NewPlayer) :-
    NewMaxM is min(10, MaxM + 1),
    NewMana = NewMaxM,
    draw_card(Deck, Hand, NewDeck, NewHand),
    % ставим CanAttack = true для всех миньонов на доске (снятие сумон-сикнесса)
    set_minions_ready(Board, ReadyBoard),
    NewPlayer = player(H, NewMana, NewMaxM, NewDeck, NewHand, ReadyBoard).

% display helpers
display_game(game_state(P1, P2, Current, Turn)) :-
    format('Turn ~w, Current player: ~w~n', [Turn, Current]),
    display_player('Player 1', P1),
    display_player('Player 2', P2).

display_player(Name, player(H, M, MaxM, Deck, Hand, Board)) :-
    format('~s: Health=~w, Mana=~w/~w~n', [Name, H, M, MaxM]),
    format('  Hand: ~w~n', [Hand]),
    format('  Board:~n'),
    display_board(Board),
    length(Deck, L), format('  Deck: ~w cards~n', [L]).

display_board([]).
display_board([minion(Name,Attack,Health,CanAttack)|T]) :-
    format('    ~w (ATK:~w HP:~w CanAttack:~w)~n', [Name, Attack, Health, CanAttack]),
    display_board(T).

% check_win_condition
check_win_condition(game_state(P1, P2, _, _), Winner) :-
    P1 = player(H1, _, _, _, _, _),
    P2 = player(H2, _, _, _, _, _),
    ( H1 =< 0 -> Winner = 2 ; H2 =< 0 -> Winner = 1 ; fail ).

% process_command wrapper
process_command(play(Minion), State, NewState) :- play_minion(State, Minion, NewState).
process_command(attack(Attacker, Target), State, NewState) :- attack(State, Attacker, Target, NewState).
process_command(end_turn, State, NewState) :- end_turn(State, NewState).

% game loop (консольный)
game_loop(State) :-
    display_game(State),
    ( check_win_condition(State, Winner) ->
        format('Player ~w wins!~n', [Winner])
    ;
        format('Enter command (play(Name), attack(AttIdx, TargetIdx), end_turn):~n'),
        read(Command),
        ( process_command(Command, State, NewState) ->
            game_loop(NewState)
        ;
            format('Invalid command or cannot perform action.~n'),
            game_loop(State)
        )
    ).

% Для запуска:
start :- initial_state(State), game_loop(State).

% -----------------------
% ПРАВИЛА / ЛОГИЧЕСКИЕ ВЫВОДЫ
% -----------------------
strong_minion(Name) :-
    ( card(Name, _, Attack, Health) -> true
    ; card_attack(Name, Attack), card_health(Name, Health)
    ),
    ( Attack >= 4 ; Health >= 5 ).

has_ability(Name, Ability) :- ability_of(Name, Ability).
is_taunt(Name) :- ability_of(Name, taunt).

% playable_card(PlayerAtom, CardName) - пример (использует in_hand/2 и player_mana/2)
player_mana(player1, 3).
player_mana(player2, 2).

playable_card(PlayerAtom, CardName) :-
    in_hand(PlayerAtom, CardName),
    ( card(CardName, Cost, _, _) -> true ; card_cost(CardName, Cost) ),
    player_mana(PlayerAtom, Mana),
    Mana >= Cost.

can_attack_hero_after_play(Card) :-
    has_ability(Card, charge).

vulnerable_minion(Name) :-
    ( card(Name, _, _, H) -> true ; card_health(Name, H) ),
    H =< 2.

% -----------------------
% ПРИМЕРЫ ЗАПРОСОВ
% -----------------------
% ?- consult('lab1_hearthstone_charge_rush.pl').
% ?- initial_state(S), display_game(S).
% ?- initial_state(S), play_minion(S, stonetusk_boar, S2), display_game(S2).
% ?- initial_state(S), play_minion(S, stonetusk_boar, S2), attack(S2, 1, 0, S3), display_game(S3).
% (последний пример: stonetusk_boar имеет ability charge -> CanAttack = true -> можно атаковать героя сразу)
%
% Для rush:
% добавь в руку карту с ability rush (например murloc_raider), сыграй и попытайся атаковать героя - должна быть ошибка
% ?- initial_state(S), play_minion(S, murloc_raider, S2), attack(S2, 1, 0, S3). % этот вызов должен fail, потому что rush не может атаковать героя
% ?- initial_state(S), play_minion(S, murloc_raider, S2), attack(S2, 1, 1, S3). % если у врага есть minion на позиции 1, допустимо

% -----------------------
% Конец файла
% -----------------------
