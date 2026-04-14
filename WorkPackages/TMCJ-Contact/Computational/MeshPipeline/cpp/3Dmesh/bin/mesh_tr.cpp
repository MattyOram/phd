
// ALTERED TO READ PARAMS FROM JSON - (SEE BOTTOM OF CGAL BOX)

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_polyhedron_3.h>
#include <CGAL/Polyhedral_complex_mesh_domain_3.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/make_mesh_3.h>


#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_segment_primitive_3.h>

#include <unordered_map>
#include <algorithm>
#include <cmath>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <filesystem>

// nlohmann/json (header-only)
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ---- CGAL types ----
using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Polyhedron = CGAL::Mesh_polyhedron_3<K>::type;

using Primitive = CGAL::AABB_face_graph_triangle_primitive<Polyhedron>;
using Traits    = CGAL::AABB_traits_3<K, Primitive>;
using Tree      = CGAL::AABB_tree<Traits>;


using Segment = K::Segment_3;
using SegIter = std::vector<Segment>::const_iterator;
using SegPrim = CGAL::AABB_segment_primitive_3<K, SegIter>;
using SegTree = CGAL::AABB_tree<CGAL::AABB_traits_3<K, SegPrim>>;


using Mesh_domain = CGAL::Polyhedral_complex_mesh_domain_3<K>;

// Triangulation / C3t3
using Tr = CGAL::Mesh_triangulation_3<Mesh_domain>::type;
using C3t3 = CGAL::Mesh_complex_3_in_triangulation_3<
  Tr,
  Mesh_domain::Corner_index,
  Mesh_domain::Curve_index
>;
using Mesh_criteria = CGAL::Mesh_criteria_3<Tr>;

namespace params = CGAL::parameters;

// size field
struct BoneCartField
{
  using Point_3 = K::Point_3;
  using Index   = Mesh_domain::Index;
  using FT      = double;

  const Mesh_domain* domain = nullptr;

  // AABB trees
  const Tree*   iface_tree      = nullptr; // interface surface
  const Tree*   cart_surf_tree = nullptr; // cartilage outer surface
  const SegTree* cart_boundary_tree = nullptr; // cartilage boundary curve (edge loop)

  // subdomain ids (must match JSON patches)
  int sd_bone = 1;
  int sd_cart = 2;

  // patch indices (must match JSON patches patch order)
  int sp_bone_surf = 1;
  int sp_cart_surf = 2;
  int sp_interface  = 3;

  // Cartilage sizing params
  FT  n_tets     = 2.0;    // number of tetrahedrons accross thickness of cartilage
  FT  min_size   = 0.02; // min target edge length (or circumradius?) within main cartilage region
  FT  max_size   = 0.50; // min target edge length (or circumradius?) within main cartilage region

  // size linearly increases from d_taper to cartilage boundary
  FT  d_taper    = 1.50; // width of cartilage taper region (region that doesn't use height based size)
  FT  taper_size = 0.20;  // target max edge length (or circumradius?) at cartilage boundary

  // Bone sizing params
  FT h_bone_max = 1.00; // max edge length (or circumradius?) - bone surface/volumetric mesh
  FT d0         = 6.0; // distance of growth region from interface edge length (or circumradius?) to h_bone_max

  // -------- helpers --------
  static FT clamp01(FT x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
  static FT lerp(FT a, FT b, FT t) { return a + (b - a) * t; }

  // Smoothstep: C1-continuous blend 0->1
  static FT smoothstep01(FT t) {
    t = clamp01(t);
    return t * t * (FT(3) - FT(2) * t);
  }

  FT dist_to_iface(const Point_3& p) const {
    return std::sqrt(iface_tree->squared_distance(p));
  }
  FT dist_to_cart_surf(const Point_3& p) const {
    return std::sqrt(cart_surf_tree->squared_distance(p));
  }
  FT dist_to_cart_boundary(const Point_3& p) const {
    return std::sqrt(cart_boundary_tree->squared_distance(p));
  }

  // "Main" cartilage size dictated by height proxy, with min clamp.
  // For volume: d_iface + d_outer
  // For interface: d_iface = 0
  // For cartilage outer: d_outer = 0
  FT cartilage_main_size(FT d_iface, FT d_outer) const {
    const FT h = d_iface + d_outer;
    const FT target = h / (std::max)(FT(1), n_tets);
    return std::clamp(target, min_size, max_size);
  }

  // Smoothed cartilage size: main sizing in interior; within taper band smoothly blend from taper_size.
  FT cartilage_size_smoothed(const Point_3& p, FT d_iface, FT d_outer) const {
    const FT main = cartilage_main_size(d_iface, d_outer);

    const FT d_bnd = dist_to_cart_boundary(p);
    if (d_bnd >= d_taper) return main;

    // Blend only inside taper region: boundary -> taper_size, at d_taper -> main
    //const FT t = smoothstep01(d_bnd / d_taper); // not linear eases in and out - makes some bad cells much worse?
    const FT t = clamp01(d_bnd / d_taper); // linear
    return lerp(taper_size, main, t);
  }

  // Bone size: ramp from local value (matching cartilage at the interface nearby) up to h_bone_max.
  FT bone_size_smoothed(const Point_3& p) const {
    const FT d = dist_to_iface(p);

    // Closest point on interface to set the local base size
    const Point_3 q = iface_tree->closest_point(p);

    // At interface, cartilage formula uses d_iface = 0, d_outer = dist_to_cart_surf(q),
    // and includes taper smoothing via distance-to-boundary at q.
    const FT base = cartilage_size_smoothed(q, /*d_iface=*/0.0, /*d_outer=*/dist_to_cart_surf(q));

    // Smoothly ramp from base -> h_bone_max over [0, d0]
    //const FT t = smoothstep01(d / d0); // not linear eases in and out
    const FT t = clamp01(d / d0); // linear

    if (base >= h_bone_max) return base; // degenerate safeguard
    return lerp(base, h_bone_max, t);
  }

  FT operator()(const Point_3& p, int dim, const Index& index) const
  {
    // 1D features (boundary curves): use the taper value
    if (dim == 1) {
      return taper_size;
    }

    if (dim == 2) {
      const int sp_i = int(domain->surface_patch_index(index));

      if (sp_i == sp_interface) {
        // interface: d_iface = 0
        return cartilage_size_smoothed(p, /*d_iface=*/0.0, /*d_outer=*/dist_to_cart_surf(p));
      }
      if (sp_i == sp_cart_surf) {
        // cartilage outer: d_outer = 0
        return cartilage_size_smoothed(p, /*d_iface=*/dist_to_iface(p), /*d_outer=*/0.0);
      }
      if (sp_i == sp_bone_surf) {
        return bone_size_smoothed(p);
      }
      // fallback
      return cartilage_size_smoothed(p, dist_to_iface(p), 0.0);
    }

    if (dim == 3) {
      const int sd = int(domain->subdomain_index(index));

      if (sd == sd_cart) {
        return cartilage_size_smoothed(p, dist_to_iface(p), dist_to_cart_surf(p));
      }
      if (sd == sd_bone) {
        return bone_size_smoothed(p);
      }
      return h_bone_max;
    }

    return taper_size;
  }
};



// facet distance
struct FacetDistanceField
{
  using Point_3 = K::Point_3;
  using Index   = Mesh_domain::Index;
  using FT      = double;

  const Mesh_domain* domain = nullptr;

  // optional trees
  const SegTree* cart_boundary_tree = nullptr;

  // patch indices - (match JSON patch order)
  int sp_bone_surf = 1;
  int sp_cart_surf = 2;
  int sp_interface  = 3;

  // constants
  // FT fd_interface = 0.01;  // interface patch (strict)
  FT fd_bone      = 1.0;  // bone patch (looser)
  FT fd_edge_loop = 0.10;  // boundary of cartilage

  // cartilage facet-distance params
  FT fd_cart_near = 0.10;  // target max facet distance near cartilage boundary
  FT fd_cart_far  = 0.05;  // target max facet distance in central cartilage region
  FT d_taper      = 1.50;  // // copied from sizing field
  FT d0_cart      = 1.5;   // width of transition band ending at d_taper

  static FT clamp01(FT x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

  FT operator()(const Point_3& p, int dim, const Index& index) const
  {
    if (dim == 1) return fd_edge_loop; // assumes boundary edge is only 1D element

    const int sp = int(domain->surface_patch_index(index));

    if (sp == sp_bone_surf) return fd_bone;

    if (sp == sp_cart_surf || sp == sp_interface) {
      const FT d = std::sqrt(cart_boundary_tree->squared_distance(p));

      // If no transition width is given, switch sharply at d_taper
      if (d0_cart <= FT(0)) {
        return (d < d_taper) ? fd_cart_near : fd_cart_far;
      }

      const FT d_start = std::max(FT(0), d_taper - d0_cart);

      // Near-boundary region
      if (d <= d_start) {
        return fd_cart_near;
      }

      // Central region
      if (d >= d_taper) {
        return fd_cart_far;
      }

      // Transition region: [d_taper - d0_cart, d_taper]
      const FT t = clamp01((d - d_start) / d0_cart);
      return fd_cart_near + (fd_cart_far - fd_cart_near) * t;
    }

    return fd_cart_far;
  }
};



static bool read_off_polyhedron(const fs::path& p, Polyhedron& poly)
{
  std::ifstream in(p);
  if(!in) return false;
  in >> poly;
  return in.good() && !poly.empty();
}

int main(int argc, char** argv)
{
  std::cout << std::unitbuf;
  if (argc != 4) {
    std::cerr
      << "Usage:\n"
      << "  " << argv[0] << " <patches.json> <params.json> <out.mesh>\n"
      << "e.g.:\n"
      << "  " << argv[0] << " TR-input/patches.json TR-input/params.json TR-output/out.mesh\n";
    return 1;
  }

  fs::path patches_path = argv[1];
  fs::path params_path  = argv[2];
  fs::path out_mesh     = argv[3];

  const fs::path base_dir = patches_path.parent_path();
  fs::create_directories(out_mesh.parent_path());

  // --- Read JSON ---
  json j;
  {
    std::ifstream jin(patches_path);
    if(!jin) {
      std::cerr << "ERROR: Cannot open JSON: " << patches_path << "\n";
      return 1;
    }
    jin >> j;
  }

  json jp;
  {
    std::ifstream pin(params_path);
    if(!pin) {
      std::cerr << "ERROR: Cannot open params JSON: " << params_path << "\n";
      return 1;
    }
    pin >> jp;
  }

  // subdomains name->int
  if(!j.contains("subdomains") || !j["subdomains"].is_object()) {
    std::cerr << "ERROR: JSON missing object 'subdomains'\n";
    return 1;
  }

  std::unordered_map<std::string, int> sub;
  for(auto it = j["subdomains"].begin(); it != j["subdomains"].end(); ++it) {
    sub[it.key()] = it.value().get<int>();
  }

  // patches array
  if(!j.contains("patches") || !j["patches"].is_array()) {
    std::cerr << "ERROR: JSON missing array 'patches'\n";
    return 1;
  }

  std::vector<Polyhedron> patches;
  std::vector<std::pair<int,int>> incident;

  // NEW: patch name -> 1-based id in read order
  std::unordered_map<std::string, int> patch_id;

  for(const auto& p : j["patches"])
  {
    const std::string name = p.value("name", "");
    const std::string file = p.value("file", "");
    if(name.empty() || file.empty()) {
      std::cerr << "ERROR: Each patch needs 'name' and 'file'\n";
      return 1;
    }

    // NEW: assign 1-based patch id based on read order
    const int this_patch_id = int(patches.size()) + 1;
    if(patch_id.find(name) != patch_id.end()) {
      std::cerr << "ERROR: Duplicate patch name: " << name << "\n";
      return 1;
    }
    patch_id[name] = this_patch_id;

    if(!p.contains("incident_subdomains") || !p["incident_subdomains"].is_array() ||
      p["incident_subdomains"].size() != 2) {
      std::cerr << "ERROR: Patch '" << name << "' needs incident_subdomains: [a,b]\n";
      return 1;
    }

    // incident_subdomains can be ["bone","outside"] or [1,0]
    auto a0 = p["incident_subdomains"][0];
    auto b0 = p["incident_subdomains"][1];

    auto resolve = [&](const json& x) -> int {
      if(x.is_number_integer()) return x.get<int>();
      if(x.is_string()) {
        auto it = sub.find(x.get<std::string>());
        if(it == sub.end()) {
          throw std::runtime_error("Unknown subdomain name: " + x.get<std::string>());
        }
        return it->second;
      }
      throw std::runtime_error("incident_subdomains entries must be int or string");
    };

    int a, b;
    try {
      a = resolve(a0);
      b = resolve(b0);
    } catch(const std::exception& e) {
      std::cerr << "ERROR in patch '" << name << "': " << e.what() << "\n";
      return 1;
    }

    const fs::path patch_path = base_dir / file;

    Polyhedron poly;
    if(!read_off_polyhedron(patch_path, poly)) {
      std::cerr << "ERROR: Failed to read OFF patch '" << name << "': " << patch_path << "\n";
      return 1;
    }

    patches.push_back(std::move(poly));
    incident.emplace_back(a, b);

    std::cout << "Loaded patch '" << name << "' (id " << this_patch_id << ") from " << patch_path
              << " with incident (" << a << "," << b << ")\n";
  }

  auto require_patch_id = [&](const std::string& name) -> int {
    auto it = patch_id.find(name);
    if (it == patch_id.end()) {
      throw std::runtime_error("Missing patch named '" + name + "'");
    }
    return it->second; // 1-based
  };

  auto patch_index0 = [&](const std::string& name) -> std::size_t {
    return std::size_t(require_patch_id(name) - 1); // 0-based vector index
  };




  // Interface surface AABB tree - for growth of cell/facet size in bone with d from interface
  const Polyhedron& interface_poly = patches[patch_index0("interface_surf")];
  Tree interface_tree(faces(interface_poly).first, faces(interface_poly).second, interface_poly);
  interface_tree.accelerate_distance_queries();

  // Cartilage boundary edge AABB tree - for growth of facet distance in cartilage surface with d from boundary
  const Polyhedron& cartilage_poly = patches[patch_index0("cartilage_surf")];
  // Cartilage outer surface AABB tree - for cartilage height sizing
  Tree cart_surf_tree(faces(cartilage_poly).first, faces(cartilage_poly).second, cartilage_poly);
  cart_surf_tree.accelerate_distance_queries();

  // Cartilage boundary for cartilage boundary distance tree
  std::vector<Segment> cartilage_boundary_segments;
  cartilage_boundary_segments.reserve(1000); // allocate memory for n segments - not needed

  for(auto h = cartilage_poly.halfedges_begin(); h != cartilage_poly.halfedges_end(); ++h) {
  if(h->is_border()) {
      const auto& p0 = h->vertex()->point();
      const auto& p1 = h->opposite()->vertex()->point();
      cartilage_boundary_segments.emplace_back(p0, p1);
  }
  }

  // AABB tree over boundary segments
  SegTree cart_bnd_tree(cartilage_boundary_segments.begin(), cartilage_boundary_segments.end());
  cart_bnd_tree.accelerate_distance_queries();


  // --- Build domain ---
  Mesh_domain domain(patches.begin(), patches.end(), incident.begin(), incident.end());
  domain.detect_borders(); // detects boundary edges that connect the separate surfaces


  // SIZING FIELD //
  BoneCartField field;
  field.domain             = &domain;
  field.iface_tree         = &interface_tree;
  field.cart_surf_tree    = &cart_surf_tree;
  field.cart_boundary_tree = &cart_bnd_tree;

  // subdomain ids from patches.json
  field.sd_bone = sub.at("bone");
  field.sd_cart = sub.at("cartilage");
  // patch ids from patches.json
  try {
    field.sp_bone_surf = require_patch_id("bone_surf");
    field.sp_cart_surf = require_patch_id("cartilage_surf");
    field.sp_interface  = require_patch_id("interface_surf");
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }

  // ---- Sizing_field params ----
  if(!jp.contains("sizing_field") || !jp["sizing_field"].is_object()) {
    std::cerr << "ERROR: params JSON missing object 'sizing_field'\n";
    return 1;
  }
  const json& s = jp["sizing_field"];

  field.n_tets     = s.value("n_tets",     field.n_tets); // number of tetrahedrons accross thickness of cartilage
  field.min_size   = s.value("min_size",   field.min_size); // min target edge length (or circumradius?) within main cartilage region
  field.max_size   = s.value("max_size",   field.max_size); // max target edge length (or circumradius?) within main cartilage region

  // edge size linearly increases from d_taper to cartilage boundary
  field.d_taper    = s.value("d_taper",    field.d_taper); // width of cartilage taper region (region that doesn't use height based size)
  field.taper_size = s.value("taper_size", field.taper_size); // target max edge length (or circumradius?) at cartilage boundary

  // bone ramp - bone surface/volume mesh grows with distance from interface
  field.h_bone_max = s.value("h_bone_max", field.h_bone_max); // max edge length (or circumradius?) - bone surface/volumetric mesh
  field.d0         = s.value("d0",         field.d0); // distance of growth region from interface edge length (or circumradius?) to h_bone_max


  // FACET DISTANCE //
  FacetDistanceField fd;
  fd.domain = &domain;
  fd.cart_boundary_tree = &cart_bnd_tree;

  // patch ids from patches.json
  fd.sp_bone_surf = field.sp_bone_surf;
  fd.sp_cart_surf = field.sp_cart_surf;
  fd.sp_interface  = field.sp_interface;

  // ---- Facet_distance params ----
  if(!jp.contains("facet_distance") || !jp["facet_distance"].is_object()) {
    std::cerr << "ERROR: params JSON missing object 'facet_distance'\n";
    return 1;
  }
  const json& f = jp["facet_distance"];

  // Allowable deviation from original mesh
  fd.fd_bone      = f.value("fd_bone",      fd.fd_bone);      // target max facet distance - bone
  fd.fd_edge_loop = f.value("fd_edge_loop", fd.fd_edge_loop); // target max facet distance - edge loop

  fd.fd_cart_near = f.value("fd_cart_near", fd.fd_cart_near); // target max facet distance near cartilage boundary
  fd.fd_cart_far  = f.value("fd_cart_far",  fd.fd_cart_far);  // target max facet distance in central cartilage region
  fd.d0_cart      = f.value("d0_cart",      fd.d0_cart);      // width of transition band ending at d_taper
  // Reuse d_taper from sizing field
  fd.d_taper      = field.d_taper;

  // MESH CRITERIA //
  // ---- criteria_params ----
  if(!jp.contains("criteria") || !jp["criteria"].is_object()) {
    std::cerr << "ERROR: params JSON missing object 'criteria'\n";
    return 1;
  }
  const json& cr = jp["criteria"];

  const double facet_angle_deg = cr.value("facet_angle", 30.0);
  const double cell_rer        = cr.value("cell_radius_edge_ratio", 3.0);

  // mesh criteria
  Mesh_criteria criteria(
    params::facet_angle(facet_angle_deg)          // target min dihedral(?) angle
                     .facet_size(field)           // target edge length of surface cells?
                     .edge_size(field)            // boundary loop of cartilage
                     .cell_size(field)            // target edge length (or circumradius?) of inner cells
                     .facet_distance(fd)          // max deviation from initial mesh surface
                     .cell_radius_edge_ratio(cell_rer) // target max radius ratio
  );

  // GENERATE MESH //
  // ---- mesh params ----
  const json& opt = jp.value("optimisation", json::object());

  const bool use_odt     = opt.value("odt", false);
  const bool use_lloyd   = opt.value("lloyd", false);
  const bool use_perturb = opt.value("perturb", true);
  const bool use_exude   = opt.value("exude", true);
  const bool return_manifold = cr.value("manifold_with_boundary", false); // this one comes from criteria

  // return manifold volume shells
  auto manifold_param =
    return_manifold
      ? params::manifold_with_boundary()
      : params::non_manifold();

  // odt
  if(!jp.contains("odt") || !jp["odt"].is_object()) {
    std::cerr << "ERROR: params JSON missing object 'odt'\n";
    return 1;
  }
  const json& oj = jp["odt"];

  const double odt_time_limit   = oj.value("time_limit", 0);
  const int    odt_max_iter     = oj.value("max_iteration_number", 0);
  const double odt_conv         = oj.value("convergence", 0.02);
  const bool   odt_do_freeze    = oj.value("do_freeze", true);
  const double odt_freeze_bound = oj.value("freeze_bound", 0.01);

  // lloyd
  if(!jp.contains("lloyd") || !jp["lloyd"].is_object()) {
    std::cerr << "ERROR: params JSON missing object 'lloyd'\n";
    return 1;
  }
  const json& lj = jp["lloyd"];

  const double lloyd_time_limit   = lj.value("time_limit", 0);
  const int    lloyd_max_iter     = lj.value("max_iteration_number", 0);
  const double lloyd_conv         = lj.value("convergence", 0.02);
  const bool   lloyd_do_freeze    = lj.value("do_freeze", true);
  const double lloyd_freeze_bound = lj.value("freeze_bound", 0.01);

  // perturb
  if(!jp.contains("perturb") || !jp["perturb"].is_object()) {
    std::cerr << "ERROR: params JSON missing object 'perturb'\n";
    return 1;
  }
  const json& pj = jp["perturb"];

  const double perturb_time_limit   = pj.value("time_limit", 0);
  const double perturb_sliver_bound = pj.value("sliver_bound", 0);

  // exude
  if(!jp.contains("exude") || !jp["exude"].is_object()) {
    std::cerr << "ERROR: params JSON missing object 'exude'\n";
    return 1;
  }
  const json& ej = jp["exude"];

  const double exude_time_limit   = ej.value("time_limit", 0);
  const double exude_sliver_bound = ej.value("sliver_bound", 0);

  // --- Generate mesh ---
  std::cout << "[[capture]] Meshing...\n";
  C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(
    domain,
    criteria,
    manifold_param, 
    params::no_odt(),
    params::no_lloyd(),
    params::no_perturb(),
    params::no_exude()
  );

  if(use_odt) {
    std::cout << "[[capture]] ODT...\n";
    const auto odt_rc = CGAL::odt_optimize_mesh_3(
      c3t3,
      domain,
      params::time_limit(odt_time_limit)
        .max_iteration_number(odt_max_iter)
        .convergence(odt_conv)
        .do_freeze(odt_do_freeze)
        .freeze_bound(odt_freeze_bound)
    );
    std::cout << "[[capture]] ODT return code: " << odt_rc << "\n";
  }

  if(use_lloyd) {
    std::cout << "[[capture]] Lloyd...\n";
    const auto lloyd_rc = CGAL::lloyd_optimize_mesh_3(
      c3t3,
      domain,
      params::time_limit(lloyd_time_limit)
        .max_iteration_number(lloyd_max_iter)
        .convergence(lloyd_conv)
        .do_freeze(lloyd_do_freeze)
        .freeze_bound(lloyd_freeze_bound)
    );
    std::cout << "[[capture]] Lloyd return code: " << lloyd_rc << "\n";
  }

  if(use_perturb) {
    std::cout << "[[capture]] Perturb...\n";
    const auto perturb_rc = CGAL::perturb_mesh_3(
      c3t3,
      domain,
      params::time_limit(perturb_time_limit)
        .sliver_bound(perturb_sliver_bound)
    );
    std::cout << "[[capture]] Perturb return code: " << perturb_rc << "\n";
  }

  if(use_exude) {
    std::cout << "[[capture]] Exude...\n";
    const auto exude_rc = CGAL::exude_mesh_3(
      c3t3,
      params::time_limit(exude_time_limit)
        .sliver_bound(exude_sliver_bound)
    );
    std::cout << "[[capture]] Exude return code: " << exude_rc << "\n";
  }

  std::cout << "Done. Writing: " << out_mesh << "\n";

  // MEDIT output
  std::ofstream out(out_mesh);
  if(!out) {
    std::cerr << "ERROR: Cannot open output: " << out_mesh << "\n";
    return 1;
  }
  c3t3.output_to_medit(out);

  std::cout << "[[capture]] Finished\n";
  std::cout << "Wrote " << out_mesh << "\n";
  return 0;
}
