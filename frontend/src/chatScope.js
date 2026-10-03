export function defaultCatalogKey(program, programs) {
  const selectedProgram = programs.find((item) => item.program_code === program);
  const editions = Array.isArray(selectedProgram?.editions)
    ? selectedProgram.editions.filter(
        (edition) => typeof edition.catalog_key === "string" && edition.catalog_key
      )
    : [];

  if (editions.length === 1) return editions[0].catalog_key;
  if (editions.length < 1) return "";

  const years = editions.map((edition) => Number(edition.academic_year));
  if (years.some((year) => !Number.isInteger(year))) return "";
  if (new Set(years).size !== years.length) return "";

  return editions[years.indexOf(Math.max(...years))].catalog_key;
}

export function resetContextForEdition(program, catalogKey) {
  return {
    ...(program ? { program } : {}),
    ...(catalogKey ? { catalog_key: catalogKey } : {}),
  };
}

export function buildConversationContext(session) {
  if (!session) return null;
  const context = { ...(session.context || {}) };
  if (session.program) context.program = session.program;

  const selectedCatalog = session.catalogKey;
  if (selectedCatalog) {
    const resultHasOtherEdition = (context.result_courses || []).some(
      (course) =>
        course.catalog_key && course.catalog_key !== selectedCatalog
    );
    if (
      (context.catalog_key && context.catalog_key !== selectedCatalog) ||
      resultHasOtherEdition
    ) {
      return resetContextForEdition(session.program, selectedCatalog);
    }
    context.catalog_key = selectedCatalog;
  }

  return Object.keys(context).length > 0 ? context : null;
}
