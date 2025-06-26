import { RoomBattleOptions } from '../room-battle';
import { Users } from '../users';
import { Rooms } from '../rooms';
import type { Connection } from '../users'; // Import Connection type

// test function to print hello world
export function helloWorld() {
    console.log('Hello, world!');
}

// const num_mcts_iterations = 1000;

// const training = Config.training || false;
// console.log(`Training mode is ${training ? 'enabled' : 'disabled'}.`);

// // Create a dummy connection object to satisfy the User constructor.
// const dummyConnection = {} as Connection; // Replace 'any' with 'Connection' if you have the type imported

// const p1 = new Users.User(dummyConnection);
// p1.name = 'MCTS Player 1';

// const p2 = new Users.User(dummyConnection);
// p2.name = 'MCTS Player 2';

// const roomBattleOptions: RoomBattleOptions = {
//     format: 'gen9randombattle',
//     players: [
//         {
//             user: p1,
//             team: ''
//         },
//         {
//             user: p2,
//             team: ''
//         },
//     ],
// };

// const battle = Rooms.createBattle(roomBattleOptions);

// Execute training functions when module is loaded
console.log('Training module loaded - executing training functions...');
helloWorld();