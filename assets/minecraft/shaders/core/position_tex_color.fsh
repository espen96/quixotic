#version 150
#moj_import <fog.glsl>
#moj_import <matrix.glsl>
#moj_import <utils.glsl>

uniform sampler2D Sampler0;
uniform sampler2D Sampler1;
uniform vec4 ColorModulator;
uniform vec2 ScreenSize;
uniform mat4 ProjMat;
in vec2 texCoord0;
in vec4 vertexColor;

uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;
uniform mat4 ModelViewMat;

out vec4 fragColor;
in vec4 texProj0;

const vec3[] COLORS = vec3[](
    vec3(0.022087, 0.098399, 0.110818),
    vec3(0.011892, 0.095924, 0.089485),
    vec3(0.027636, 0.101689, 0.100326),
    vec3(0.046564, 0.109883, 0.114838),
    vec3(0.064901, 0.117696, 0.097189),
    vec3(0.063761, 0.086895, 0.123646),
    vec3(0.084817, 0.111994, 0.166380),
    vec3(0.097489, 0.154120, 0.091064),
    vec3(0.106152, 0.131144, 0.195191),
    vec3(0.097721, 0.110188, 0.187229),
    vec3(0.133516, 0.138278, 0.148582),
    vec3(0.070006, 0.243332, 0.235792),
    vec3(0.196766, 0.142899, 0.214696),
    vec3(0.047281, 0.315338, 0.321970),
    vec3(0.204675, 0.390010, 0.302066),
    vec3(0.080955, 0.314821, 0.661491)
);

const mat4 SCALE_TRANSLATE = mat4(
    0.5, 0.0, 0.0, 0.25,
    0.0, 0.5, 0.0, 0.25,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0
);

mat4 end_portal_layer(float layer) {
    mat4 translate = mat4(
        1.0, 0.0, 0.0, 17.0 / layer,
        0.0, 1.0, 0.0, (2.0 + layer / 1.5) * (GameTime * 1.5),
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    mat2 rotate = mat2_rotate_z(radians((layer * layer * 4321.0 + layer * 9.0) * 2.0));

    mat2 scale = mat2((4.5 - layer / 4.0) * 2.0);

    return mat4(scale * rotate) * translate * SCALE_TRANSLATE;
}


void main() {
    int index = inControl(gl_FragCoord.xy, ScreenSize.x);
    if (index != -1) {
                 
    int index = inControl(gl_FragCoord.xy, ScreenSize.x);
    if (index != -1) {
   
            // store ProjMat in control pixels
            if (index >= 5 && index <= 15) {
                int c = (index - 5) / 4;
                int r = (index - 5) - c * 4;
                c = (c == 0 && r == 1) ? c : c + 1;
                fragColor = vec4(encodeFloat(ProjMat[c][r]), 1.0);
            }
            // store ModelViewMat in control pixels
            else if (index >= 16 && index <= 24) {
                int c = (index - 16) / 3;
                int r = (index - 16) - c * 3;
                fragColor = vec4(encodeFloat(ModelViewMat[c][r]), 1.0);
            }
            // store ProjMat[0][0] and ProjMat[1][1] in control pixels
            else if (index >= 3 && index <= 4) {
                fragColor = vec4(encodeFloat(atan(ProjMat[index - 3][index - 3])), 1.0);
            }  
            // store FogColor in control pixels
            else if (index == 25) {
                fragColor = vec4(FogColor.rgb,clamp(FogEnd/255,0,1));
            
            }  
            // store SkyColor? in control pixels
            else if (index == 26) {
                fragColor = vec4(ColorModulator.rgb,1);
            } 
                        // store SkyColor? in control pixels
            else if (index == 27) {
                fragColor = vec4(vec3(GameTime),1);
            }     
            else if (index == 29) {
                fragColor = vec4(1.0);
            }                             
            // blackout control pixels for sunDir so sun can write to them (by default, all pixels are FogColor)
            else {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }
  
    }   
    } else {
        vec4 color = texture(Sampler0, texCoord0) * vertexColor;
        if (color.a < 0.1) {
            discard;
        }
        fragColor = color * ColorModulator;
            float far = ProjMat[3][2] / (ProjMat[2][2] + 1);
    vec3 color2 = textureProj(Sampler0, texProj0).rgb * COLORS[0];
    for (int i = 0; i < 1; i++) {
        color2 += textureProj(Sampler1, texProj0 * end_portal_layer(float(i + 1))).rgb * COLORS[i];
    }
    color2 *= vec3(1.0,0.2,0.5);

        if(far > 9.99996)fragColor.rgb = color2;
    }
}
