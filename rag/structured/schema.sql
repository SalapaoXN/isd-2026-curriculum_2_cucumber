PRAGMA foreign_keys = ON;

CREATE TABLE provenance (
    provenance_id INTEGER PRIMARY KEY,
    source_document_key TEXT NOT NULL,
    program TEXT,
    source_filename TEXT,
    source_page INTEGER,
    document_page INTEGER,
    document_category TEXT NOT NULL DEFAULT 'unknown'
        CHECK (document_category IN ('plan', 'description', 'unknown')),
    source_uri TEXT,
    source_locator TEXT,
    excerpt TEXT
);

CREATE TABLE catalogs (
    catalog_id INTEGER PRIMARY KEY,
    catalog_key TEXT,
    title TEXT,
    academic_year TEXT,
    institution TEXT,
    notes TEXT
);

CREATE TABLE programs (
    program_id INTEGER PRIMARY KEY,
    catalog_id INTEGER NOT NULL REFERENCES catalogs(catalog_id),
    program_code TEXT NOT NULL,
    program_code_normalized TEXT NOT NULL,
    UNIQUE (catalog_id, program_code_normalized)
);

CREATE TABLE courses (
    course_id INTEGER PRIMARY KEY,
    catalog_id INTEGER NOT NULL REFERENCES catalogs(catalog_id),
    course_code TEXT NOT NULL,
    course_code_normalized TEXT NOT NULL,
    name_th TEXT,
    name_en TEXT,
    credits TEXT,
    credit_units INTEGER,
    credits_raw TEXT,
    description_th TEXT,
    description_en TEXT,
    category TEXT,
    course_type TEXT,
    prerequisite_text TEXT,
    notes TEXT,
    UNIQUE (catalog_id, course_code_normalized)
);

CREATE TABLE curriculum_plans (
    plan_id INTEGER PRIMARY KEY,
    catalog_id INTEGER NOT NULL REFERENCES catalogs(catalog_id),
    program_id INTEGER NOT NULL REFERENCES programs(program_id),
    program_code TEXT NOT NULL,
    plan_key TEXT NOT NULL,
    plan_code TEXT,
    plan_name TEXT,
    version TEXT,
    notes TEXT,
    UNIQUE (catalog_id, program_id, plan_key)
);

CREATE TABLE alternative_course_groups (
    alternative_group_id INTEGER PRIMARY KEY,
    catalog_id INTEGER NOT NULL REFERENCES catalogs(catalog_id),
    plan_id INTEGER REFERENCES curriculum_plans(plan_id),
    group_key TEXT,
    label TEXT,
    minimum_choices INTEGER NOT NULL DEFAULT 1 CHECK (minimum_choices >= 1),
    maximum_choices INTEGER NOT NULL DEFAULT 1
        CHECK (maximum_choices >= minimum_choices),
    notes TEXT
);

CREATE TABLE alternative_course_group_members (
    alternative_group_member_id INTEGER PRIMARY KEY,
    alternative_group_id INTEGER NOT NULL
        REFERENCES alternative_course_groups(alternative_group_id)
        ON DELETE CASCADE,
    course_id INTEGER NOT NULL REFERENCES courses(course_id),
    member_order INTEGER,
    notes TEXT,
    UNIQUE (alternative_group_id, course_id)
);

CREATE TABLE plan_placements (
    placement_id INTEGER PRIMARY KEY,
    plan_id INTEGER NOT NULL REFERENCES curriculum_plans(plan_id)
        ON DELETE CASCADE,
    course_id INTEGER REFERENCES courses(course_id),
    alternative_group_id INTEGER
        REFERENCES alternative_course_groups(alternative_group_id),
    year_number INTEGER,
    semester_number INTEGER,
    flexible_year_number INTEGER,
    flexible_semester_number INTEGER,
    flexible_year_semester_raw TEXT,
    category TEXT,
    requirement_type TEXT,
    placement_order INTEGER,
    credits_override TEXT,
    raw_text TEXT,
    notes TEXT,
    CHECK (
        (course_id IS NOT NULL AND alternative_group_id IS NULL)
        OR
        (course_id IS NULL AND alternative_group_id IS NOT NULL)
    )
);

CREATE TABLE prerequisites (
    prerequisite_id INTEGER PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES courses(course_id)
        ON DELETE CASCADE,
    prerequisite_course_id INTEGER REFERENCES courses(course_id),
    alternative_group_id INTEGER
        REFERENCES alternative_course_groups(alternative_group_id),
    prerequisite_order INTEGER,
    requirement_type TEXT NOT NULL DEFAULT 'required',
    raw_text TEXT,
    notes TEXT,
    CHECK (
        (prerequisite_course_id IS NOT NULL AND alternative_group_id IS NULL)
        OR
        (prerequisite_course_id IS NULL AND alternative_group_id IS NOT NULL)
        OR
        (prerequisite_course_id IS NULL AND alternative_group_id IS NULL
            AND raw_text IS NOT NULL)
    )
);

CREATE TABLE catalog_provenance (
    catalog_id INTEGER NOT NULL REFERENCES catalogs(catalog_id)
        ON DELETE CASCADE,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
        ON DELETE CASCADE,
    PRIMARY KEY (catalog_id, provenance_id)
);

CREATE TABLE course_provenance (
    course_id INTEGER NOT NULL REFERENCES courses(course_id)
        ON DELETE CASCADE,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
        ON DELETE CASCADE,
    source_order INTEGER,
    PRIMARY KEY (course_id, provenance_id)
);

CREATE TABLE curriculum_plan_provenance (
    plan_id INTEGER NOT NULL REFERENCES curriculum_plans(plan_id)
        ON DELETE CASCADE,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
        ON DELETE CASCADE,
    PRIMARY KEY (plan_id, provenance_id)
);

CREATE TABLE plan_placement_provenance (
    placement_id INTEGER NOT NULL REFERENCES plan_placements(placement_id)
        ON DELETE CASCADE,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
        ON DELETE CASCADE,
    PRIMARY KEY (placement_id, provenance_id)
);

CREATE TABLE prerequisite_provenance (
    prerequisite_id INTEGER NOT NULL REFERENCES prerequisites(prerequisite_id)
        ON DELETE CASCADE,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
        ON DELETE CASCADE,
    PRIMARY KEY (prerequisite_id, provenance_id)
);

CREATE TABLE alternative_group_provenance (
    alternative_group_id INTEGER NOT NULL
        REFERENCES alternative_course_groups(alternative_group_id)
        ON DELETE CASCADE,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
        ON DELETE CASCADE,
    PRIMARY KEY (alternative_group_id, provenance_id)
);

CREATE TABLE alternative_group_member_provenance (
    alternative_group_member_id INTEGER NOT NULL
        REFERENCES alternative_course_group_members(alternative_group_member_id)
        ON DELETE CASCADE,
    provenance_id INTEGER NOT NULL REFERENCES provenance(provenance_id)
        ON DELETE CASCADE,
    PRIMARY KEY (alternative_group_member_id, provenance_id)
);

CREATE INDEX courses_by_catalog_and_code
    ON courses (catalog_id, course_code);

CREATE INDEX plans_by_catalog
    ON curriculum_plans (catalog_id);

CREATE INDEX programs_by_catalog_and_code
    ON programs (catalog_id, program_code);

CREATE INDEX placements_by_plan
    ON plan_placements (plan_id, year_number, semester_number, placement_order);

CREATE INDEX placements_by_course
    ON plan_placements (course_id);

CREATE INDEX prerequisites_by_course
    ON prerequisites (course_id, prerequisite_order);

CREATE INDEX alternative_members_by_group
    ON alternative_course_group_members (alternative_group_id, member_order);

CREATE INDEX alternative_members_by_course
    ON alternative_course_group_members (course_id);

CREATE VIEW v_plan_courses AS
SELECT
    plans.plan_id,
    plans.program_id,
    programs.program_code AS program,
    COALESCE(plans.plan_code, plans.plan_key) AS plan,
    plans.plan_code,
    plans.plan_key,
    placements.placement_id,
    placements.placement_order,
    placements.year_number AS year,
    placements.semester_number AS semester,
    placements.flexible_year_number,
    placements.flexible_semester_number,
    placements.flexible_year_semester_raw,
    COALESCE(placements.course_id, members.course_id) AS course_id,
    COALESCE(courses.course_code, member_courses.course_code) AS course,
    COALESCE(courses.course_code, member_courses.course_code) AS course_code,
    COALESCE(courses.credit_units, member_courses.credit_units) AS credits,
    COALESCE(courses.credit_units, member_courses.credit_units) AS credit_units,
    COALESCE(courses.credits_raw, member_courses.credits_raw) AS credits_raw,
    placements.alternative_group_id,
    members.member_order AS alternative_member_order,
    groups.minimum_choices,
    groups.maximum_choices,
    CASE WHEN placements.alternative_group_id IS NULL THEN 0 ELSE 1 END
        AS is_alternative,
    placements.category,
    placements.requirement_type,
    placements.credits_override,
    placements.raw_text,
    placements.notes
FROM plan_placements AS placements
JOIN curriculum_plans AS plans
    ON plans.plan_id = placements.plan_id
JOIN programs
    ON programs.program_id = plans.program_id
LEFT JOIN courses
    ON courses.course_id = placements.course_id
LEFT JOIN alternative_course_groups AS groups
    ON groups.alternative_group_id = placements.alternative_group_id
LEFT JOIN alternative_course_group_members AS members
    ON members.alternative_group_id = placements.alternative_group_id
LEFT JOIN courses AS member_courses
    ON member_courses.course_id = members.course_id;

CREATE VIEW v_semester_credits AS
WITH counted_courses AS (
    SELECT *
    FROM v_plan_courses
    WHERE alternative_group_id IS NULL
       OR (
           alternative_member_order IS NOT NULL
           AND alternative_member_order <= minimum_choices
       )
)
SELECT
    plan_id,
    program,
    plan,
    year,
    semester,
    COALESCE(SUM(credit_units), 0) AS total_credits
FROM counted_courses
GROUP BY plan_id, program, plan, year, semester;

CREATE VIEW v_prerequisite_edges AS
SELECT
    prerequisites.prerequisite_id,
    source.course_id AS source_course_id,
    source.course_code AS source_course,
    source.course_code AS source_course_code,
    COALESCE(prerequisites.prerequisite_course_id, members.course_id)
        AS prerequisite_course_id,
    target.course_code AS prerequisite_course,
    target.course_code AS prerequisite_code,
    prerequisites.alternative_group_id,
    members.member_order AS alternative_member_order,
    prerequisites.prerequisite_order,
    prerequisites.requirement_type,
    prerequisites.raw_text
FROM prerequisites
JOIN courses AS source
    ON source.course_id = prerequisites.course_id
LEFT JOIN alternative_course_group_members AS members
    ON members.alternative_group_id = prerequisites.alternative_group_id
LEFT JOIN courses AS target
    ON target.course_id = COALESCE(
        prerequisites.prerequisite_course_id, members.course_id
    );
