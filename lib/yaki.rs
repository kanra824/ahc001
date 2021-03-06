use proconio::input;
use std::time::Instant;

const TIME_LIMIT: u128 = 1900;
const LOOP_PER_TIME_CHECK: usize = 100;
const START_TMP: f64 = 50.0;
const END_TMP: f64 = 10.0;


#[derive(Debug)]
#[allow(dead_code)]
pub struct Xorshift {
    seed: u64,
}
impl Xorshift {
    #[allow(dead_code)]
    pub fn new() -> Xorshift {
        Xorshift {
            seed: 0xf0fb588ca2196dac,
        }
    }
    #[allow(dead_code)]
    pub fn with_seed(seed: u64) -> Xorshift {
        Xorshift { seed: seed }
    }
    #[inline]
    #[allow(dead_code)]
    pub fn next(&mut self) -> u64 {
        self.seed = self.seed ^ (self.seed << 13);
        self.seed = self.seed ^ (self.seed >> 7);
        self.seed = self.seed ^ (self.seed << 17);
        self.seed
    }
    #[inline]
    #[allow(dead_code)]
    pub fn rand(&mut self, m: u64) -> u64 {
        self.next() % m
    }
    #[inline]
    #[allow(dead_code)]
    pub fn rand_int(&mut self, min: u64, max: u64) -> u64 {
        self.next() % (max - min + 1) + min
    }
    #[inline]
    #[allow(dead_code)]
    // 0.0 ~ 1.0
    pub fn randf(&mut self) -> f64 {
        use std::mem;
        const UPPER_MASK: u64 = 0x3FF0000000000000;
        const LOWER_MASK: u64 = 0xFFFFFFFFFFFFF;
        let tmp = UPPER_MASK | (self.next() & LOWER_MASK);
        let result: f64 = unsafe { mem::transmute(tmp) };
        result - 1.0
    }
}

struct State {
    rand: Xorshift,
    score: i64,
}
impl State {
    fn new(rand: Xorshift) -> Self {
        let mut state = State{rand: rand, score: 0};

        state.score = state.score_all();

        state
    }

    fn update(&mut self, temperature: f64) {
        //let prob = std::f64::consts::E.powf((new_score - self.score) as f64 / temperature);
        // !!!!!!!!!!!!!!!!!!!! MAXIMIZE !!!!!!!!!!!!!!!!!!!!

        //if self.rand.randf() < prob {
            // update
        //}
    }

    fn score_all(&mut self) -> i64 {
        0
    }
}

fn simulated_annealing(state: &mut State, start: &Instant) {
    let mut elapsed_time = start.elapsed().as_millis();
    while elapsed_time < TIME_LIMIT {
        eprintln!("{}", elapsed_time);
        for _ in 0..LOOP_PER_TIME_CHECK {
            let temperature: f64 = START_TMP + (END_TMP - START_TMP) * (elapsed_time as f64) / (TIME_LIMIT as f64);
            state.update(temperature);
        }
        elapsed_time = start.elapsed().as_millis();
    }
}

fn main() {
    let start = Instant::now();

    let seed = start.elapsed().as_nanos() as u64;
    let rand = Xorshift::with_seed(seed);
    let mut state = State::new(rand);
    simulated_annealing(&mut state, &start);

    // print score
    // print answer

}