#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D temporals3Sampler;
uniform vec2 OutSize;
uniform vec2 ScreenSize;
uniform float Time;
uniform mat4 ProjMat;

in vec3 flareColor;

in float GameTime;
in vec2 texCoord;
in vec2 texCoord2;
in vec2 oneTexel;
in vec3 sunDir;
in vec4 fogcol;
in vec4 skycol;
in vec4 rain;
in mat4 gbufferModelViewInverse;
in mat4 gbufferModelView;
in mat4 gbufferProjection;
in mat4 gbufferProjectionInverse;
in float near;
in float far;

in float aspectRatio;
in float cosFOVrad;
in float tanFOVrad;
in mat4 gbPI;
in mat4 gbP;






out vec4 fragColor;

// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define NUMCONTROLS 26
#define THRESH 0.5
#define FPRECISION 4000000.0
#define PROJNEAR 0.05
#define FUDGE 32.0




vec3 toLinear(vec3 sRGB){
	return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}



float rgb2lumi(vec4 color) {
    return dot(color.rgb, vec3(0.212, 0.716, 0.072));
    //return color.g;
}

vec2 gradientAt(sampler2D tex, vec2 pos, vec2 sampling)
{   
    vec2 stepx = vec2(sampling.x, 0.0);
    vec2 stepy = vec2(0.0, sampling.y);
    
    /* Explicit Sobel
    float left = rgb2lumi( texture(tex, pos-stepx) ) * 2.0
               + rgb2lumi( texture(tex, pos-stepx-stepy) )
               + rgb2lumi( texture(tex, pos-stepx+stepy) );
    float right = rgb2lumi( texture(tex, pos+stepx) ) * 2.0
                + rgb2lumi( texture(tex, pos+stepx-stepy) )
                + rgb2lumi( texture(tex, pos+stepx+stepy) );
    
    float top = rgb2lumi( texture(tex, pos-stepy) ) * 2.0
              + rgb2lumi( texture(tex, pos-stepy-stepx) )
              + rgb2lumi( texture(tex, pos-stepy+stepx) );
    float bottom = rgb2lumi( texture(tex, pos+stepy) ) * 2.0
                 + rgb2lumi( texture(tex, pos+stepy-stepx) )
                 + rgb2lumi( texture(tex, pos+stepy+stepx) );
    */
    
    // 3x3 derivative kernel operator making use of bilinear interpolation.
    // The sobel operator is not rotationally invariant at all. Scharr is 
    // better. Kroon did some work in finding an optimal kernel as well.
    // The center pixel has an effective coefficient of (1-alpha)*2
    // The outer pixels have an effective coefficient of alpha
    // To calculate alpha from the relation between the two:
    // alpha = 2 / (2+R)
    // Sobel (R=2):         alpha = 0.5
    // Scharr (R=10/3):     alpha = 0.375
    // Kroon (R=61/17):     alpha = 0.358
    
    float alpha = 0.358;
    float left = rgb2lumi( texture(tex, pos-stepx - alpha*stepy) )
               + rgb2lumi( texture(tex, pos-stepx + alpha*stepy) );
    float right = rgb2lumi( texture(tex, pos+stepx - alpha*stepy) )
                + rgb2lumi( texture(tex, pos+stepx + alpha*stepy) );
    
    float top = rgb2lumi( texture(tex, pos-stepy - alpha*stepx) ) 
              + rgb2lumi( texture(tex, pos-stepy + alpha*stepx) );
    float bottom = rgb2lumi( texture(tex, pos+stepy - alpha*stepx) ) 
                 + rgb2lumi( texture(tex, pos+stepy + alpha*stepx) );

   
    // 5x5 derivative kernel operator making use of bilinear interpolation.
    // Took me a while to calculate these coefficients. I hope I did not make 
    // a mistake. In the end, it turned out not too work better than the 3x3
    // kernel. The diffusion should be in the gradient field, not the original
    // albeido image.
    /*
    float a1 = 1.12;  // 0.12; 0.04
    float a2 = 1.06;  // 0.06; 0.03
    float a3 = 1.11157;  // 0.11157; 0.125  relation between two center elements
    float a4 = 2.3472;  // 2.3472; 2.3063  relation between center and corner
    float left = rgb2lumi( texture(tex, pos - a3*stepx) ) * a4
               + rgb2lumi( texture(tex, pos - a1*stepx - a2*stepy) )
               + rgb2lumi( texture(tex, pos - a1*stepx + a2*stepy) );
    float right = rgb2lumi( texture(tex, pos + a3*stepx) ) * a4
                + rgb2lumi( texture(tex, pos + a1*stepx - a2*stepy) )
                + rgb2lumi( texture(tex, pos + a1*stepx + a2*stepy) );  
    float top = rgb2lumi( texture(tex, pos - a3*stepy) ) * a4
                 + rgb2lumi( texture(tex, pos - a1*stepy - a2*stepx) )
                 + rgb2lumi( texture(tex, pos - a1*stepy + a2*stepx) );
    float bottom = rgb2lumi( texture(tex, pos + a3*stepy) ) * a4
                 + rgb2lumi( texture(tex, pos + a1*stepy - a2*stepx) )
                 + rgb2lumi( texture(tex, pos + a1*stepy + a2*stepx) );
    */
    
    // Return gradient
    //return vec2(right-left, bottom-top) / (a4+2.0);
    return vec2(right-left, bottom-top) * 0.5;
}


void main() {
    float aspectRatio = ScreenSize.x/ScreenSize.y;
    vec4 screenPos = gl_FragCoord;
         screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
         screenPos.zw = vec2(1.0);
    vec3 view = normalize((gbufferModelViewInverse * screenPos).xyz);


    float diffuseDepth = texture(DiffuseDepthSampler, texCoord).r;
    vec3 OutTexel = texture(DiffuseSampler, texCoord).rgb;


    vec2 shape = ScreenSize;
    vec2 sampling = 1.0/ shape;
    vec2 stepx = vec2(sampling.x, 0.0);
    vec2 stepy = vec2(0.0, sampling.y);
    
    // Get centre location
    vec2 pos = texCoord;
    
    // Calculate gradient
    vec2 grad = gradientAt(DiffuseSampler, pos, sampling);
    
    // Calculate normal from gradient, and its magnitude
    vec2 N = vec2(-grad.y, grad.x);
    float gm = length(N);
    
    // Stop if the edge is weak
    if ( gm < 0.05) {
        fragColor.rgb = texture(DiffuseSampler, pos).rgb;
        fragColor.a = 1.0;
        return;
    }
    
    // Refine the calculation of the gradient, by looking at the surroundings
    // This corresponds to a Gaussian with sigma = 1.0.
    // Note that this is diffusion of the gradient vector field, *not* of the
    // albeido image.
    // I'm not even sure whether this significantly improves the result, but
    // improvement may be more apparent if we use a fix for near 
    // horizontal/vertical edges.
    float alpha1 = 0.6;
    float alpha2 = 0.37;
    
    grad += gradientAt(DiffuseSampler, pos -stepx, sampling) * alpha1;
    grad += gradientAt(DiffuseSampler, pos +stepx, sampling) * alpha1;
    grad += gradientAt(DiffuseSampler, pos -stepy, sampling) * alpha1;
    grad += gradientAt(DiffuseSampler, pos +stepy, sampling) * alpha1;
    grad += gradientAt(DiffuseSampler, pos -stepx -stepy, sampling) * alpha2;
    grad += gradientAt(DiffuseSampler, pos -stepx +stepy, sampling) * alpha2;
    grad += gradientAt(DiffuseSampler, pos +stepx -stepy, sampling) * alpha2;
    grad += gradientAt(DiffuseSampler, pos +stepx +stepy, sampling) * alpha2;
    
    // Recalculate gradient
    N = vec2(-grad.y, grad.x);
    N = normalize(N);
    
    // todo: how to determine strength, and extend ???
    // We can probably gain some quality by tweaking these in the right way.
    float strength = 0.25;
    float extend = 1.5;
    
    // Initialize result with original value
    vec3 result = texture(DiffuseSampler, pos).rgb * (1.0-strength);
    
    // todo: correct for undersampling at near-horizontal and near-vertical edges
    
    // Diffuse with samples along the edge
    // todo: what coefficients to use?
    N = extend * N;
    result += texture(DiffuseSampler, pos+N*sampling).rgb * strength * 0.5;
    result += texture(DiffuseSampler, pos-N*sampling).rgb * strength * 0.5;
    //result += texture(DiffuseSampler, pos+N*sampling).rgb * strength * 0.333;
    //result += texture(DiffuseSampler, pos-N*sampling).rgb * strength * 0.333;
    //result += texture(DiffuseSampler, pos+N*2.0*sampling).rgb * strength * 0.16666;
    //result += texture(DiffuseSampler, pos-N*2.0*sampling).rgb * strength * 0.16666;
    
    // Set result
    fragColor.rgb = result;
    fragColor.a = 1.0;
}
