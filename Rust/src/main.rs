#[macro_use]
extern crate glium;

mod cloth_classes;
use cloth_classes::Cloth;
use glium::{glutin::GL_CORE, vertex};

const NUM_THREADS: usize = 8;

fn main() {
    // Create cloth
    let mut cloth = Cloth::new(1000, 1000, true, NUM_THREADS);

    // OpenGL setup
    #[allow(unused_imports)]
    use glium::{glutin, Surface};

    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new();
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    #[derive(Copy, Clone)]
    struct Vertex {
        position: [f32; 2],
    }

    implement_vertex!(Vertex, position);

    let point1 = Vertex { position: [0.0, 0.0] };
    let point2 = Vertex { position: [ 0.05,  0.0] };
    let shape = vec![point1, point2];

    let vertex_buffer = glium::VertexBuffer::new(&display, &shape).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::LineStrip);

    let vertex_shader_src = r#"
        #version 140

        in vec2 position;

        uniform mat4 matrix_transform;
        uniform mat4 matrix_scale;
        uniform mat4 matrix_rotation;

        void main() {
            gl_Position =  matrix_transform * matrix_rotation * matrix_scale * vec4(position, 0.0, 1.0);
        }
    "#;

    let fragment_shader_src = r#"
        #version 140

        out vec4 color;

        void main() {
            color = vec4(0.0, 0.0, 0.0, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();
    let mut sim_times = vec![];
    let mut start_time = std::time::Instant::now();
    let mut end_time = std::time::Instant::now();
    event_loop.run(move |event, _, control_flow| {
        let delta_t = start_time.duration_since(end_time).as_secs_f32();
        start_time = std::time::Instant::now();
        match event {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                _ => return,
            },
            glutin::event::Event::NewEvents(cause) => match cause {
                glutin::event::StartCause::ResumeTimeReached { .. } => (),
                glutin::event::StartCause::Init => (),
                
                _ => return,
            },
            _ => return,
        }

        let next_frame_time = std::time::Instant::now() + std::time::Duration::from_nanos(16_666_667);
        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);

        // Begin render loop
        

        // Create a drawing target
        let mut target = display.draw();

        // Clear the screen to black
        target.clear_color(1.0, 1.0, 1.0, 1.0);

        // Iterate over the springs
        for i in 0..cloth.springs.len() {

            let node1 = cloth.springs[i].point1.lock().unwrap();
            let node2 = cloth.springs[i].point2.lock().unwrap();
            
            let translation_matrix = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [(node1.x * 0.05) - 0.8, (node1.y * 0.05) - 0.5, 0.0, 1.0]
            ];

            let dif_x = node2.x - node1.x;
            let dif_y = node2.y - node1.y;
            let length = (dif_x*dif_x + dif_y*dif_y).sqrt();

            let scale_matrix = [
                [length, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ];
            
            // calculate angle of line where horizontal is 0
            let angle = -dif_y.atan2(dif_x);
            // make rotation matrix
            let rotation_matrix = [
                [angle.cos(), -angle.sin(), 0.0, 0.0],
                [angle.sin(), angle.cos(), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ];

            let uniforms = uniform! {
                matrix_transform: translation_matrix,
                matrix_scale: scale_matrix,
                matrix_rotation: rotation_matrix,
            };

            // Draw the triangle
            target.draw(&vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();
            end_time = std::time::Instant::now();
        }

        let start = std::time::Instant::now();
        cloth.run_sim(delta_t);
        sim_times.push(std::time::Instant::now().duration_since(start).as_secs_f32());
        if sim_times.len() > 100 {
            let mut sum = 0.0;
            for i in 0..sim_times.len() {
                sum += sim_times[i];
            }
            println!("Average simulation time: {}", sum / sim_times.len() as f32);
            sim_times.clear();
        }

        // Display the completed drawing
        target.finish().unwrap();
        // End render loop

        
    });
}

