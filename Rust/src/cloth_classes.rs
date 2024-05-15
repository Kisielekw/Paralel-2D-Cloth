use std::sync::{Arc, Mutex};

pub struct Node{
    pub x: f32,
    pub y: f32,
    velocity: [f32; 2],
    acceleration: [f32; 2],
    mass: f32,
    dampening: f32,
    can_move: bool,
    springs: Vec<Arc<Spring>>,
}
impl Node{
    pub fn new(x: f32, y: f32, can_move: bool) -> Node{
        Node{
            x: x,
            y: y,
            velocity: [0.0, 0.0],
            acceleration: [0.0, 0.0],
            mass: 0.01,
            dampening: 0.03,
            can_move: can_move,
            springs: Vec::new(),
        }
    }

    fn apply_force(&mut self, force: [f32; 2]){
        self.acceleration[0] += force[0] / self.mass;
        self.acceleration[1] += force[1] / self.mass;
    }

    pub fn apply_gravity(&mut self){
        self.apply_force([0.0, -9.81 * self.mass]);
    }

    pub fn update(&mut self, dt: f32){
        if self.can_move{
            let mut dampening_force = [0.0, 0.0];
            dampening_force[0] = -self.velocity[0] * self.dampening;
            dampening_force[1] = -self.velocity[1] * self.dampening;

            self.apply_force(dampening_force);

            let new_x = self.x + self.velocity[0] * dt + 0.5 * self.acceleration[0] * dt * dt;
            let new_y = self.y + self.velocity[1] * dt + 0.5 * self.acceleration[1] * dt * dt;

            if(dt > 0.0){
                self.velocity[0] = (new_x - self.x) / dt;
                self.velocity[1] = (new_y - self.y) / dt;
            }
            else{
                self.velocity[0] = 0.0;
                self.velocity[1] = 0.0;
            }

            self.x = new_x;
            self.y = new_y;
        }
        
        self.acceleration = [0.0, 0.0];
    }

    pub fn add_spring(&mut self, spring: Arc<Spring>){
        self.springs.push(spring);
    }
}

pub struct Spring{
    pub point1: Arc<Mutex<Node>>,
    pub point2: Arc<Mutex<Node>>,
    rest_length: f32,
    spring_coe: f32,
}

impl Spring{
    pub fn new(point1: Arc<Mutex<Node>>, point2: Arc<Mutex<Node>>) -> Spring{
        Spring{
            point1: point1,
            point2: point2,
            rest_length: 1.0,
            spring_coe: 10.0,
        }
    }

    pub fn calc_forces(&self){
        let mut point1 = self.point1.lock().unwrap();
        let mut point2 = self.point2.lock().unwrap();

        let dx = point2.x - point1.x;
        let dy = point2.y - point1.y;

        let dist = (dx*dx + dy*dy).sqrt();

        let force = (dist - self.rest_length) * self.spring_coe;

        let force_x = (force * dx) / dist;
        let force_y = (force * dy) / dist;

        point1.apply_force([force_x, force_y]);
        point2.apply_force([-force_x, -force_y]);
    }
}

pub struct Cloth{
    nodes: Vec<Arc<Mutex<Node>>>,
    pub springs: Vec<Arc<Spring>>,
    apply_gravity: bool,
    thread_number: usize,
}
impl Cloth{
    pub fn new(rows: u32, cols: u32, apply_gravity: bool, thread_number: usize) -> Cloth{
        let mut nodes = vec![];
        for y in 0..rows {
            for x in 0..cols {
                if (x ==  0 || x == cols - 1) && y == rows - 1 {
                    nodes.push(Arc::new(Mutex::new(Node::new(x as f32, y as f32, false))));
                    continue;
                }

                nodes.push(Arc::new(Mutex::new(Node::new(x as f32, y as f32, true))));
            }
        }

        let mut springs = vec![];
        for y in 0..rows {
            for x in 0..cols {
                if x < cols - 1 {
                    springs.push(Arc::new(Spring::new(nodes[(y*cols + x) as usize].clone(), nodes[(y*cols + x + 1) as usize].clone())));
                    nodes[(y*cols+x) as usize].lock().unwrap().add_spring(springs[springs.len()-1].clone());
                    nodes[(y*cols+x+1) as usize].lock().unwrap().add_spring(springs[springs.len()-1].clone());
                }
                if y < rows - 1 {
                    springs.push(Arc::new(Spring::new(nodes[(y*cols + x) as usize].clone(), nodes[((y+1)*cols + x) as usize].clone())));
                    nodes[(y*cols+x) as usize].lock().unwrap().add_spring(springs[springs.len()-1].clone());
                    nodes[((y+1)*cols+x) as usize].lock().unwrap().add_spring(springs[springs.len()-1].clone());
                }
            }
        }

        Cloth{
            nodes: nodes,
            springs: springs,
            apply_gravity: apply_gravity,
            thread_number: thread_number,
        }
    }

    pub fn _toggle_gravity(&mut self,){
        self.apply_gravity = self.apply_gravity != true;
    }

    pub fn run_sim(&mut self, dt: f32){
        let mut pool = scoped_threadpool::Pool::new(self.thread_number as u32);

        // Update the velocity of the nodes
        pool.scoped(|scope|{
            for slice in self.springs.chunks(self.springs.len() / self.thread_number){
                scope.execute(move || {
                    for spring in slice{
                        spring.calc_forces();
                    }
                });
            }
        });

        // Update the position of the nodes
        pool.scoped(|scope|{
            let local_gravity = self.apply_gravity;
            for slice in self.nodes.chunks(self.nodes.len() / self.thread_number){
                scope.execute(move || {
                    for node in slice{
                        let mut lock = node.lock().unwrap();
                        
                        if local_gravity {
                            lock.apply_gravity();
                        }

                        lock.update(dt);
                    }
                });
            }
        });
    }
}